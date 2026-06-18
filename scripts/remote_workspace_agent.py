from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import signal
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

CODE_DIRS = ("src", "scripts", "configs", "doc", "tests")
CODE_FILES = ("README.md", "SKILL.md", "pyproject.toml", "requirements.txt", ".gitignore")
MAX_BODY_BYTES = 128 * 1024 * 1024
MAX_SYNC_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_SYNC_FILES = 50_000
ACTIVE_JOB_STATUSES = ("queued", "starting", "running")


class TLSThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], context: ssl.SSLContext) -> None:
        super().__init__(server_address, handler_class)
        self.context = context

    def get_request(self) -> tuple[ssl.SSLSocket, tuple[str, int]]:
        while True:
            raw_socket, address = self.socket.accept()
            try:
                return self.context.wrap_socket(raw_socket, server_side=True), address
            except ssl.SSLError as exc:
                print(f"TLS handshake failed from {address[0]}:{address[1]}: {exc}", file=sys.stderr, flush=True)
                raw_socket.close()


def background_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def now() -> float:
    return time.time()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes workspace root: {path}")
    return path


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return f'"{pid}"'.encode("ascii") in (result.stdout or b"")
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created REAL NOT NULL,
                    started REAL,
                    finished REAL,
                    pid INTEGER,
                    timeout_seconds INTEGER NOT NULL,
                    exit_code INTEGER,
                    payload_json TEXT NOT NULL,
                    stdout_path TEXT NOT NULL,
                    stderr_path TEXT NOT NULL,
                    error TEXT
                )
                """
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def create(self, kind: str, payload: dict[str, Any], timeout_seconds: int, job_dir: Path) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        row = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created": now(),
            "started": None,
            "finished": None,
            "pid": None,
            "timeout_seconds": timeout_seconds,
            "exit_code": None,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "stdout_path": str(job_dir / job_id / "stdout.log"),
            "stderr_path": str(job_dir / job_id / "stderr.log"),
            "error": None,
        }
        Path(row["stdout_path"]).parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.connect() as db:
            db.execute(
                """INSERT INTO jobs VALUES (:id,:kind,:status,:created,:started,:finished,:pid,:timeout_seconds,
                   :exit_code,:payload_json,:stdout_path,:stderr_path,:error)""",
                row,
            )
        return self.get(job_id)

    def update(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.lock, self.connect() as db:
            db.execute(f"UPDATE jobs SET {assignments} WHERE id=?", [*values.values(), job_id])

    def claim(self, job_id: str) -> bool:
        with self.lock, self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='starting', started=? WHERE id=? AND status='queued'",
                (now(), job_id),
            )
            return cursor.rowcount == 1

    def finish_if_running(self, job_id: str, **values: Any) -> bool:
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.lock, self.connect() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {assignments} WHERE id=? AND status='running'",
                [*values.values(), job_id],
            )
            return cursor.rowcount == 1

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["alive"] = process_alive(result.get("pid"))
        return result

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock, self.connect() as db:
            rows = db.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self.get(row["id"]) for row in rows]

    def ids_with_status(self, statuses: tuple[str, ...]) -> list[str]:
        placeholders = ",".join("?" for _ in statuses)
        with self.lock, self.connect() as db:
            rows = db.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders}) ORDER BY created",
                statuses,
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def reconcile(self) -> None:
        with self.lock, self.connect() as db:
            rows = db.execute("SELECT id,pid FROM jobs WHERE status IN ('starting','running')").fetchall()
        for row in rows:
            if process_alive(row["pid"]):
                kill_tree(row["pid"])
            self.update(
                row["id"],
                status="interrupted",
                finished=now(),
                exit_code=-1,
                error="workspace agent restarted while job was active",
            )


class WorkspaceState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.root.resolve()
        self.repo = inside(self.root, args.repo)
        self.runs = inside(self.root, args.runs)
        self.python = args.python.resolve()
        self.token = args.token
        self.jobs_root = inside(self.root, args.jobs_root)
        self.staging_root = inside(self.root, args.staging_root)
        for path in (self.root, self.repo, self.runs, self.jobs_root, self.staging_root):
            path.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(inside(self.root, args.database))
        self.store.reconcile()
        self.gpu_lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.sync_lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}
        for job_id in self.store.ids_with_status(("queued",)):
            self._start_thread(job_id)

    def command_for(self, job_id: str, kind: str, payload: dict[str, Any]) -> tuple[list[str], Path, bool]:
        if kind == "shell":
            script = str(payload.get("script", ""))
            if not script.strip():
                raise ValueError("shell job requires a non-empty script")
            if len(script.encode("utf-8")) > 1024 * 1024:
                raise ValueError("shell script exceeds 1 MiB")
            arguments = payload.get("arguments", [])
            if not isinstance(arguments, list) or len(arguments) > 64:
                raise ValueError("shell arguments must be a list with at most 64 values")
            arguments = [str(value) for value in arguments]
            if any(len(value) > 4096 for value in arguments):
                raise ValueError("shell argument exceeds 4096 characters")
            cwd_value = str(payload.get("cwd", "."))
            cwd = inside(self.root, self.root / cwd_value)
            if not cwd.is_dir():
                raise ValueError(f"shell cwd does not exist: {cwd}")
            script_path = inside(self.jobs_root, self.jobs_root / job_id / "script.ps1")
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script, encoding="utf-8")
            return [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                *arguments,
            ], cwd, False
        if kind == "diagnostics":
            args = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.repo / "scripts" / "workspace_diagnostics_job.ps1"),
                "-Python",
                str(self.python),
            ]
            return args, self.repo, False
        if kind == "report":
            raise ValueError("report jobs are not implemented for UDLF workspace service yet")
        if kind == "temporal_audit":
            raise ValueError("temporal_audit jobs are not implemented for UDLF workspace service yet")
        if kind == "train":
            run_name = str(payload.get("run_name") or time.strftime("workspace_%Y%m%d_%H%M%S"))
            run_dir = inside(self.runs, self.runs / run_name)
            wrapper = ["-Template", str(payload["template"]), "-RunName", run_name]
            for key, value in payload.get("params", {}).items():
                flag = "-" + str(key)
                if isinstance(value, bool):
                    if value:
                        wrapper.append(flag)
                elif value is not None:
                    wrapper.extend([flag, str(value)])
            args = [
                str(self.python),
                str(self.repo / "scripts" / "remote_workspace_train_job.py"),
                "--repo",
                str(self.repo),
                "--python",
                str(self.python),
                "--run-dir",
                str(run_dir),
                *wrapper,
            ]
            return args, self.repo, True
        if kind == "cleanup_checkpoints":
            run_name = str(payload["run_name"])
            run_dir = inside(self.runs, self.runs / run_name)
            script = (
                f"Get-ChildItem -LiteralPath '{run_dir}' -Recurse -File -Filter '*.pt' "
                "| Remove-Item -Force; Write-Output 'checkpoints removed'"
            )
            return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], self.repo, False
        raise ValueError(f"unsupported job kind: {kind}")

    def submit(self, kind: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        with self.lifecycle_lock:
            job = self.store.create(kind, payload, timeout_seconds, self.jobs_root)
            self._start_thread(job["id"])
        return job

    def _start_thread(self, job_id: str) -> None:
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        self.threads[job_id] = thread
        thread.start()

    def _run_job(self, job_id: str) -> None:
        initial = self.store.get(job_id)
        lock = self.gpu_lock if initial["kind"] in {"train", "temporal_audit"} else threading.Lock()
        with lock:
            try:
                with self.lifecycle_lock:
                    if not self.store.claim(job_id):
                        return
                    job = self.store.get(job_id)
                    args, cwd, _gpu = self.command_for(job_id, job["kind"], job["payload"])
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(self.repo / "src")
                    env["UDLF_PYTHON"] = str(self.python)
                    env["PYTHON"] = str(self.python)
                    env["PYTHONUTF8"] = "1"
                    stdout_path = Path(job["stdout_path"])
                    stderr_path = Path(job["stderr_path"])
                    stdout = stdout_path.open("ab", buffering=0)
                    stderr = stderr_path.open("ab", buffering=0)
                    try:
                        proc = subprocess.Popen(
                            args,
                            cwd=cwd,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            creationflags=background_creation_flags(),
                            start_new_session=os.name != "nt",
                        )
                    except Exception:
                        stdout.close()
                        stderr.close()
                        raise
                    self.store.update(job_id, status="running", pid=proc.pid)
                try:
                    exit_code = proc.wait(timeout=job["timeout_seconds"])
                    status = "succeeded" if exit_code == 0 else "failed"
                    self.store.finish_if_running(job_id, status=status, finished=now(), exit_code=exit_code)
                except subprocess.TimeoutExpired:
                    kill_tree(proc.pid)
                    self.store.finish_if_running(
                        job_id,
                        status="timed_out",
                        finished=now(),
                        exit_code=-1,
                        error=f"timed out after {job['timeout_seconds']} seconds",
                    )
                finally:
                    stdout.close()
                    stderr.close()
            except Exception as exc:
                self.store.update(job_id, status="failed", finished=now(), exit_code=-1, error=repr(exc))

    def stop(self, job_id: str) -> dict[str, Any]:
        with self.lifecycle_lock:
            job = self.store.get(job_id)
            if job["status"] not in ACTIVE_JOB_STATUSES:
                return job
            if job.get("pid") and process_alive(job["pid"]):
                kill_tree(job["pid"])
            self.store.update(job_id, status="cancelled", finished=now(), exit_code=-1, error="cancelled by client")
        return self.store.get(job_id)

    def apply_sync(self, archive: Path, expected_sha256: str) -> dict[str, Any]:
        with self.sync_lock, self.lifecycle_lock:
            if self.store.ids_with_status(ACTIVE_JOB_STATUSES):
                raise RuntimeError("repository sync refused while jobs are active")
            digest = sha256_file(archive)
            if not hmac.compare_digest(digest.lower(), expected_sha256.lower()):
                raise ValueError("archive sha256 mismatch")
            stage = self.staging_root / f"sync-{uuid.uuid4().hex}"
            backup = self.staging_root / f"backup-{uuid.uuid4().hex}"
            stage.mkdir(parents=True)
            try:
                with zipfile.ZipFile(archive) as source:
                    items = source.infolist()
                    files = [item for item in items if not item.is_dir()]
                    if len(files) > MAX_SYNC_FILES or sum(item.file_size for item in files) > MAX_SYNC_EXTRACTED_BYTES:
                        raise ValueError("sync archive extracted content exceeds limits")
                    for item in items:
                        target = inside(stage, stage / item.filename)
                        if item.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with source.open(item) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                if not (stage / "src").is_dir() or not (stage / "scripts").is_dir() or not (stage / "pyproject.toml").is_file():
                    raise ValueError("sync archive is missing required repository content")
                backup.mkdir()
                try:
                    for name in (*CODE_DIRS, *CODE_FILES):
                        current = self.repo / name
                        replacement = stage / name
                        if current.exists():
                            shutil.move(str(current), str(backup / name))
                        if replacement.exists():
                            shutil.move(str(replacement), str(current))
                except Exception:
                    for name in (*CODE_DIRS, *CODE_FILES):
                        current = self.repo / name
                        saved = backup / name
                        remove_path(current)
                        if saved.exists():
                            shutil.move(str(saved), str(current))
                    raise
                count = sum(1 for path in self.repo.rglob("*") if path.is_file())
                return {"sha256": digest, "files": count}
            finally:
                shutil.rmtree(stage, ignore_errors=True)
                shutil.rmtree(backup, ignore_errors=True)
                archive.unlink(missing_ok=True)


def read_chunk(path: Path, offset: int, limit: int = 1024 * 1024) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    size = path.stat().st_size
    offset = min(max(offset, 0), size)
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read(limit)
        next_offset = file.tell()
    return data.decode("utf-8", errors="replace"), next_offset


class Handler(BaseHTTPRequestHandler):
    state: WorkspaceState
    server_version = "UDLFWorkspace/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.log_date_time_string()} {fmt % args}\n")

    def authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {self.state.token}")

    def send_json(self, status: int, value: Any) -> None:
        body = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self.authorized():
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/v1/health":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "tls": True,
                        "root": str(self.state.root),
                        "repo": str(self.state.repo),
                        "runs": str(self.state.runs),
                        "jobs": self.state.store.list(5),
                    },
                )
                return
            if parsed.path == "/v1/jobs":
                self.send_json(200, {"ok": True, "jobs": self.state.store.list(int(query.get("limit", ["50"])[0]))})
                return
            if parsed.path == "/v1/files":
                relative = query.get("path", [""])[0]
                path = inside(self.state.root, self.state.root / relative)
                if not path.is_file():
                    self.send_json(404, {"ok": False, "error": "file not found"})
                    return
                size = path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Content-SHA256", sha256_file(path))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with path.open("rb") as source:
                    shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
                return
            if parsed.path.startswith("/v1/jobs/"):
                parts = parsed.path.split("/")
                job = self.state.store.get(parts[3])
                if len(parts) == 5 and parts[4] == "logs":
                    stdout, stdout_offset = read_chunk(
                        Path(job["stdout_path"]), int(query.get("stdout_offset", ["0"])[0])
                    )
                    stderr, stderr_offset = read_chunk(
                        Path(job["stderr_path"]), int(query.get("stderr_offset", ["0"])[0])
                    )
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "job": job,
                            "stdout": stdout,
                            "stderr": stderr,
                            "stdout_offset": stdout_offset,
                            "stderr_offset": stderr_offset,
                        },
                    )
                else:
                    self.send_json(200, {"ok": True, "job": job})
                return
            self.send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": repr(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        try:
            if not self.authorized():
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            parsed = urlparse(self.path)
            if parsed.path != "/v1/files":
                self.send_json(404, {"ok": False, "error": "not found"})
                return
            query = parse_qs(parsed.query)
            path = inside(self.state.root, self.state.root / query.get("path", [""])[0])
            length = int(self.headers.get("Content-Length", "0"))
            expected = self.headers.get("X-Content-SHA256", "")
            if length < 0 or length > 8 * 1024 * 1024 * 1024:
                raise ValueError("invalid file size")
            temporary = inside(self.state.staging_root, self.state.staging_root / f"file-{uuid.uuid4().hex}.tmp")
            digest = hashlib.sha256()
            with temporary.open("wb") as target:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("file upload ended early")
                    target.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual = digest.hexdigest()
            if not hmac.compare_digest(actual.lower(), expected.lower()):
                temporary.unlink(missing_ok=True)
                raise ValueError("file sha256 mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, path)
            self.send_json(200, {"ok": True, "path": str(path), "size": length, "sha256": actual})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": repr(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if not self.authorized():
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            if self.path == "/v1/jobs":
                payload = self.payload()
                timeout = min(max(int(payload.get("timeout_seconds", 3600)), 1), 7 * 24 * 3600)
                job = self.state.submit(str(payload["kind"]), dict(payload.get("payload", {})), timeout)
                self.send_json(202, {"ok": True, "job": job})
                return
            if self.path.startswith("/v1/jobs/") and self.path.endswith("/stop"):
                self.send_json(200, {"ok": True, "job": self.state.stop(self.path.split("/")[3])})
                return
            if self.path == "/v1/sync":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("invalid sync archive size")
                expected = self.headers.get("X-Content-SHA256", "")
                archive = self.state.staging_root / f"upload-{uuid.uuid4().hex}.zip"
                with archive.open("wb") as target:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("sync upload ended early")
                        target.write(chunk)
                        remaining -= len(chunk)
                self.send_json(200, {"ok": True, **self.state.apply_sync(archive, expected)})
                return
            self.send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="TLS-only persistent UDLF remote workspace agent.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9543)
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument("--token")
    token_group.add_argument("--token-file", type=Path)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.cert.is_file() or not args.key.is_file():
        raise SystemExit("TLS certificate and private key are required")
    if args.token_file:
        args.token = args.token_file.read_text(encoding="ascii").strip()
    if not args.token:
        raise SystemExit("workspace token is empty")

    Handler.state = WorkspaceState(args)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.cert, args.key)
    server = TLSThreadingHTTPServer((args.host, args.port), Handler, context)
    print(f"UDLF workspace agent listening on https://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
