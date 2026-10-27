# Runtime architecture

```text
CLI / Python / stdio MCP / authenticated HTTP MCP
                       │
                    Harness
                       │
        SQLite registry, queue, events, artifacts
                       │
             detached supervisor (flock)
                       │ one active job
                  worker process
                       │
      LangGraph: model → validated tool → model → finalize
                       │                       │
          read-only specialist pool        frozen checks
                       │                       │
               source hash binding      Bubblewrap execution
```

## Durable boundaries

Submission freezes the typed task, effective model configuration, and resolved profile. The supervisor admits one worker per state directory. A queue lock coordinates submission/resume with supervisor shutdown; worker PID ownership prevents an exiting old attempt from overwriting a newly queued resume. Client disconnection does not cancel jobs.

The registry is SQLite with WAL. `SqliteSaver` checkpoints the graph under a run/revision thread ID. A tool node interrupted by a process failure is replayed with the same action key. Edit receipts bind the entire action plus expected/desired file hashes; completed writes can be recognized after interruption, conflicting state stops replay. Started command receipts never count as completed evidence; commands rerun. A pending LangGraph `interrupt()` stores a question, and resume supplies `Command(resume=answer)` to that checkpoint.

Specialist records retain tasks, messages, source fingerprints, profiles, and model configurations. Completed matching work survives worker restart; failed or missing work resumes without repeating successful siblings. An assistant action saved before an interrupted read is replayed before another request. Worker completion must be collected before writer edits/execution. Profile changes explicitly start a new graph revision and change cache keys.

Cancellation is a durable flag checked during HTTP model requests and command polling. Workers also handle termination signals. The supervisor terminates unresponsive workers, and Bubblewrap's parent-death behavior closes execution namespaces. Command groups are killed on timeout/cancellation. Completed artifacts and unknown/started receipts remain retained.

## Source and execution boundaries

The effective baseline includes selected dirty/untracked source and modes. Git worktrees use a private index; no commit is created. Worktree creation registers metadata in the source repository's Git administration area, but the source checkout/index stay unchanged. Git metadata operations disable hooks, filesystem monitors, and configured clean/smudge/process filters to avoid executing project commands outside the sandbox. Non-Git/unborn/subdirectory selections use filtered copies. Source symlinks fail explicitly.

The broker validates every action and role capability. Scoped workers receive bounded excerpts and permitted read tools. Source evidence is content-addressed and carries path, hash, and range. Recovered citations must match current source. Semantic index output is navigation only; claims use newly read source matching the effective snapshot. Persistent index lifecycle operations are never implicit.

Bubblewrap gives project code no host-home or source-checkout mount. System toolchains and declared dependencies are read-only; workspace/temp storage is writable. `.git` is read-only inside execution. A Git worktree's original common Git directory is not exposed, so project commands that require full original Git history may fail. The environment is cleared, network is normally isolated, and process resource/output/wall-time limits apply. Profiles explicitly enable network/installation. Sandbox errors stop execution.

## Completion and evidence

Check definitions are frozen separately from model conversations. Registered new tests are frozen as immutable artifacts and installed in baseline execution snapshots as well as candidate sources. Baseline snapshots are checked for integrity; check results are bound to the full selected source/mode fingerprint. Prerequisite success must match that fingerprint and phase. Existing test/build oracles are retained and checked for bypass attempts.

Finalization executes required baseline and candidate checks, obtains a fresh scoped specialist review when enabled, validates cited source evidence, and computes a verdict from receipts. Model text cannot upgrade a missing, failed, skipped, truncated, or interrupted check. Review claims remain hypotheses; unresolved claims/gaps cannot silently become a verified result. Reports identify check phases, outcomes, hashes, and artifact IDs. A completed graph is distinct from a verified repair.

The implementation deliberately does not prove arbitrary check commands adequate, hide unsolved model behavior, apply patches to source, commit changes, deploy services, consume external MCP tools, or support multiple users/parallel writers.
