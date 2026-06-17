# UDLF Workspace

This workspace contains the UDLF language-model design, implementation scaffold,
and remote 4090 operation workflow.

## Layout

- `doc/` - design documents and imported remote-operation notes.
- `scripts/` - remote SSH, sync, monitor, detached launch, and training helper
  scripts for UDLF.
- `configs/` - workspace configuration examples and imported remote training
  templates.
- `src/udlf/` - future UDLF implementation package.
- `tests/` - future tests.
- `experiments/` - experiment definitions and lightweight records.
- `runs/` - local run outputs. Do not commit generated runs.
- `artifacts/` - generated reports, plots, or exported analysis artifacts.

## Remote 4090 Materials

The imported remote workflow is documented in `doc/REMOTE_4090_OPERATIONS.md`.
The operational scripts are intentionally preserved close to their source form
so prior workflow behavior remains auditable.

Important caveat: the main remote scripts are UDLF-oriented, but the current
training entrypoint is only a smoke runner. Do not launch a real UDLF training
run until the stage A model implementation and a real UDLF config exist.

Create private machine config from the example:

```powershell
Copy-Item configs\workspace.example.json configs\workspace.local.json
```

Never commit `configs/workspace.local.json`, real host names, credentials,
private absolute paths, checkpoints, or run outputs.
