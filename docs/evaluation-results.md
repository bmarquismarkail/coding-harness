# v0.2 implementation evaluation

The implemented service passes its deterministic execution and interface suite. **Live autonomous coding acceptance has not passed.** Infrastructure validation and model task quality are separate results.

## Deterministic suite

Command, using the requested environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/python-venv/langchain-suite/bin/python \
  -m unittest discover -s tests -v
```

Result: **48 tests discovered, 45 passed, 3 opt-in live tests skipped**, in 36.059 seconds. The retained log is `/home/user/.local/state/coding-agent-acceptance/v0.2-unittest-final.log`. `pip check` reports no broken requirements. `git diff --check` passes. `agent-config.json` and tracked `development-evidence/` have no changes.

The suite executes Python, C11, and C++17 repairs in actual Bubblewrap sandboxes. Trusted acceptance cases fail on baseline sources and pass after the deterministic model fixture proposes the patch; original regressions also run. The selected source checkout stays unchanged. Separate cases exercise malformed actions, stale evidence, frozen/original oracle bypasses, source modifications during checks, Git filters, dirty baselines, edit receipts/modes, interrupted commands, partial cancellation logs, separate budgets, specialist cache invalidation, one failed worker among nine, and supervisor restart/cancellation.

Real SDK clients exercise stdio and Streamable HTTP submission, disconnection, event pagination, result/artifact retrieval, resume, and cancellation. HTTP tests reject missing/wrong credentials, the model key, invalid hosts/origins, and repositories outside allowed roots. Batch CLI submission survives process exit, and its results match the Python API. These transport tests use a deterministic local HTTP model fixture; they do not establish live model coding quality.

The venv contains the installed editable v0.2 package, `mcp==2.3.0`, and `langgraph-checkpoint-sqlite==3.1.2`. Model identities, endpoint locations, and credential environment-variable references remain unchanged.

## Live model acceptance

Eight unassisted development attempts were made through the shared service, using the configured main and specialist endpoints. No fixture patch or final synthesis was supplied manually. Every attempt remains recorded as **failed**, with its original profile, requests, responses, checks, workspace, and artifacts under `/home/user/.local/state/coding-agent-acceptance/`.

| Fixture/evidence folder | Recorded outcome |
| --- | --- |
| `cpp-n9aqsbp4` | Invalid action envelopes; no patch |
| `python-9jwxrpnr` | Invalid actions/findings exhausted the reduced development budget |
| `cpp-8irl949_` | Repeated investigation exhausted the reduced development budget |
| `python-6hneg9y_` | Faulty repair; remaining budget exhausted during review |
| `cpp-8uqnat7r` | Correct patch and passing checks, but reviewer incorrectly claimed recorded verification was missing; result remained unverified |
| `python-6hnocive` | Incorrect repair and failing final checks; result remained unverified |
| `cpp-vjya16r5` | Faulty repair/test changes and truncated specialist output; interrupted, unverified |
| `python-xkgqew57` | Specialist action budget exhausted; interrupted, unverified |

Each folder has `acceptance.json` with the run ID, state directory, result/error, independent grading status, and observed model roles. Earlier attempts used a reduced 40-action development budget; later attempts used 200. The final implementation records tool/check actions separately from model requests. These are development iterations across code changes, not a controlled model benchmark. Failed records were not reset, manually repaired, or promoted after later implementation changes.

Live evidence establishes communication with both configured models and exposes unsuccessful autonomous task behavior. It does not establish reliable autonomous coding or C++ audit quality. Further model/workflow evaluation remains necessary before treating this harness as reliably self-sufficient.

## Scope and limitations

Runtime state is outside repositories. No changes were applied to Proto-Time, no project commit/push/PR was created, and no persistent semantic index was rebuilt. TLS deployment, external MCP consumption, multiple-user authorization, parallel writers, and a web UI remain deferred.

Automatic verification conservatively rejects original build-oracle changes. Source symlinks and files larger than the snapshot limit require materialized fixtures. Original Git history is not exposed inside project execution. Other languages can use configured command arrays, but this evaluation directly covers Python, C, and C++.

See [v0.2.md](v0.2.md) for commands and configuration, [architecture.md](architecture.md) for persistence/trust boundaries, and [mcp-evaluation.md](mcp-evaluation.md) for read-only interface questions. The implementation follows the [MCP SDK](https://github.com/modelcontextprotocol/python-sdk), [LangGraph persistence API](https://docs.langchain.com/oss/python/langgraph/persistence), and [llama-server JSON response format](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
