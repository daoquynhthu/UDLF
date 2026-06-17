# Issues

This file tracks only repeated problems, active blockers, or major risks that
materially affect execution. Fast, one-off fixes should not be recorded here.

## Active

### Stage A training harness does not exist yet

Status: open.

The remote workflow can now handle generic UDLF SSH, sync, status, log, STOP,
and detached-launch operations. A minimal `udlf.training.train` smoke entrypoint
exists, but it only validates workflow plumbing. It is not the UDLF stage A
training loop. A minimal Stage A model forward/loss now exists, but there is no
real data pipeline, optimizer loop, checkpoint policy, or intervention
evaluation yet.

Impact:

- Remote code sync and inspection can be prepared.
- Remote smoke runs can be used to validate infrastructure.
- Real UDLF model training cannot be launched yet.

Resolution direction:

- Implement the real stage A training loop and synthetic task pipeline.
- Keep smoke runs clearly labeled as infrastructure checks.


## Resolved

None yet.
