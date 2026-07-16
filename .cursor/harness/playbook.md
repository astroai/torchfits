# Harness Playbook

## Bullets

- id: verify-tiers
  desc: verify_fast during edits; verify (ci-local) before push; verify_full only when human opts in — never torchregress-harness in the agent loop.

- id: ci-parity
  desc: GitHub CI lint+test matrix; local ci-local = preflight-push + pixi test.

- id: file-memory
  desc: Put durable notes in .cursor/harness/ — not long chat scrollback.

- id: minimal-diff
  desc: Smallest correct change; match existing repo style and tools.

- id: docs-api-sync
  desc: Root table helpers return dict[str,Tensor]; Arrow path is torchfits.table.*; never document env vars absent from src; integrity tests guard Core I/O signature fences + api.md env table.
