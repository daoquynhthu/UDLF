from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import ssl
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


class Client:
    def __init__(self, host: str, port: int, token: str, fingerprint: str) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.fingerprint = fingerprint.lower().replace(":", "")

    def connect(self, timeout: int = 60) -> http.client.HTTPSConnection:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(self.host, self.port, context=context, timeout=timeout)
        conn.connect()
        certificate = conn.sock.getpeercert(binary_form=True)
        actual = hashlib.sha256(certificate).hexdigest()
        if actual != self.fingerprint:
            conn.close()
            raise RuntimeError(f"TLS certificate fingerprint mismatch: expected {self.fingerprint}, got {actual}")
        return conn

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> dict:
        conn = self.connect()
        request_headers = {"Authorization": f"Bearer {self.token}"}
        request_headers.update(headers or {})
        conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        data = json.loads(raw or b"{}")
        if response.status >= 400 or not data.get("ok", False):
            raise RuntimeError(f"workspace request failed ({response.status}): {data}")
        return data

    def upload_file(self, source: Path, destination: str) -> dict:
        size = source.stat().st_size
        digest = sha256_file(source)
        conn = self.connect(timeout=300)
        conn.putrequest("PUT", f"/v1/files?{urlencode({'path': destination})}")
        conn.putheader("Authorization", f"Bearer {self.token}")
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(size))
        conn.putheader("X-Content-SHA256", digest)
        conn.endheaders()
        with source.open("rb") as body:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                conn.send(chunk)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        data = json.loads(raw or b"{}")
        if response.status >= 400 or not data.get("ok", False):
            raise RuntimeError(f"workspace upload failed ({response.status}): {data}")
        return data

    def download_file(self, source: str, destination: Path) -> dict:
        conn = self.connect(timeout=300)
        conn.request("GET", f"/v1/files?{urlencode({'path': source})}", headers={"Authorization": f"Bearer {self.token}"})
        response = conn.getresponse()
        if response.status >= 400:
            raw = response.read()
            conn.close()
            raise RuntimeError(f"workspace download failed ({response.status}): {raw.decode(errors='replace')}")
        expected = response.getheader("X-Content-SHA256", "")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if not expected or actual.lower() != expected.lower():
                raise RuntimeError(f"workspace download sha256 mismatch: expected {expected}, got {actual}")
            os.replace(temporary, destination)
            return {"ok": True, "path": str(destination), "size": destination.stat().st_size, "sha256": actual}
        finally:
            conn.close()
            temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_sync_archive(repo: Path) -> Path:
    include_dirs = ("src", "scripts", "configs", "doc", "tests")
    include_files = (
        "README.md",
        "SKILL.md",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
        "plan.md",
        "progress.md",
    )
    excluded_dirs = {".git", ".venv312", ".pytest_cache", ".ruff_cache", "__pycache__"}
    excluded_suffixes = {".pyc", ".pt", ".pth", ".zip", ".jsonl", ".log"}
    handle, name = tempfile.mkstemp(prefix="udlf-workspace-", suffix=".zip")
    os.close(handle)
    archive = Path(name)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        for root in include_dirs:
            base = repo / root
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(repo)
                if any(part in excluded_dirs for part in rel.parts) or path.suffix.lower() in excluded_suffixes:
                    continue
                target.write(path, rel.as_posix())
        for name in include_files:
            path = repo / name
            if path.is_file():
                target.write(path, name)
    return archive


def wait_job(client: Client, job_id: str, follow: bool) -> int:
    stdout_offset = 0
    stderr_offset = 0
    while True:
        query = urlencode({"stdout_offset": stdout_offset, "stderr_offset": stderr_offset})
        data = client.request("GET", f"/v1/jobs/{job_id}/logs?{query}")
        stdout_offset = data["stdout_offset"]
        stderr_offset = data["stderr_offset"]
        if data["stdout"]:
            sys.stdout.write(data["stdout"])
            sys.stdout.flush()
        if data["stderr"]:
            sys.stderr.write(data["stderr"])
            sys.stderr.flush()
        job = data["job"]
        if job["status"] not in {"queued", "starting", "running"}:
            print(json.dumps(job, ensure_ascii=False, indent=2))
            return 0 if job["status"] == "succeeded" else 1
        if not follow:
            print(json.dumps(job, ensure_ascii=False, indent=2))
            return 0
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned-TLS client for the UDLF remote workspace agent.")
    parser.add_argument("--config", type=Path, default=Path("configs/workspace.local.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    status = sub.add_parser("status")
    status.add_argument("job_id", nargs="?")
    sub.add_parser("sync")
    push = sub.add_parser("push")
    push.add_argument("source", type=Path)
    push.add_argument("destination")
    pull = sub.add_parser("pull")
    pull.add_argument("source")
    pull.add_argument("destination", type=Path)
    job = sub.add_parser("job")
    job.add_argument(
        "kind",
        choices=["shell", "diagnostics", "train", "report", "temporal_audit", "cleanup_checkpoints"],
    )
    job.add_argument("--payload", default="{}")
    job.add_argument("--timeout", type=int, default=3600)
    job.add_argument("--no-follow", action="store_true")
    logs = sub.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--follow", action="store_true")
    stop = sub.add_parser("stop")
    stop.add_argument("job_id")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = config["remote"]["workspace_service"]
    client = Client(
        host=str(workspace["host"]),
        port=int(workspace.get("port", 9543)),
        token=str(workspace["token"]),
        fingerprint=str(workspace["certificate_sha256"]),
    )
    if args.command == "health":
        print(json.dumps(client.request("GET", "/v1/health"), ensure_ascii=False, indent=2))
    elif args.command == "status":
        path = f"/v1/jobs/{args.job_id}" if args.job_id else "/v1/jobs"
        print(json.dumps(client.request("GET", path), ensure_ascii=False, indent=2))
    elif args.command == "sync":
        archive = make_sync_archive(Path.cwd())
        try:
            body = archive.read_bytes()
            result = client.request(
                "POST", "/v1/sync", body, {"Content-Type": "application/zip", "X-Content-SHA256": hashlib.sha256(body).hexdigest()}
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            archive.unlink(missing_ok=True)
    elif args.command == "push":
        print(json.dumps(client.upload_file(args.source.resolve(), args.destination), ensure_ascii=False, indent=2))
    elif args.command == "pull":
        print(json.dumps(client.download_file(args.source, args.destination.resolve()), ensure_ascii=False, indent=2))
    elif args.command == "job":
        payload = {"kind": args.kind, "payload": json.loads(args.payload), "timeout_seconds": args.timeout}
        result = client.request("POST", "/v1/jobs", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        job_id = result["job"]["id"]
        print(f"job_id={job_id}")
        raise SystemExit(wait_job(client, job_id, not args.no_follow))
    elif args.command == "logs":
        raise SystemExit(wait_job(client, args.job_id, args.follow))
    elif args.command == "stop":
        print(json.dumps(client.request("POST", f"/v1/jobs/{args.job_id}/stop", b"{}"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
