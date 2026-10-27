# Read-only MCP evaluation corpus

Use completed, retained deterministic repair fixtures from `tests/test_v2.py` and `tests/test_v2_acceptance.py`. Supply their run IDs as `PYTHON_RUN`, `CPP_RUN`, and the review-negative run as `NEGATIVE_RUN`. These are synthetic model-response fixtures with real execution, not claims about live model reliability. Start the MCP server against that retained state and its allowed source root. Preserve state for the evaluator; do not rerun jobs while answering.

Each independent question uses only `harness_status`, `harness_events`, `harness_result`, and `harness_artifact`. The evaluator should follow event/artifact pagination and source/check artifact links. No question requires submission, resume, cancellation, source modification, or external access. The transport/lifecycle acceptance tests separately execute the mutating operations using real SDK clients.

| # | Question | Expected single answer |
| --- | --- | --- |
| 1 | For PYTHON_RUN, read its result and the baseline acceptance artifact. Which exception type demonstrates the original generator failure? | `TypeError` |
| 2 | For PYTHON_RUN, reconcile status with the final result and final acceptance artifact. What is its verification outcome? | `verified` |
| 3 | For PYTHON_RUN, page through source/tool evidence and inspect the exported patch. Which production file changes? | `latency.py` |
| 4 | For PYTHON_RUN, inspect the final patch artifact after checking its immutable ID. What counter variable is added to support single-use iterables? | `count` |
| 5 | For PYTHON_RUN, reconcile the final source hashes and check artifacts. What exit code does final candidate acceptance report? | `0` |
| 6 | For CPP_RUN, inspect the check definitions retained in the result and the baseline executable artifact. What exit code reproduces the bug? | `1` |
| 7 | For CPP_RUN, follow the candidate build/check evidence and patch pages. What upper clamp value replaces 99? | `100` |
| 8 | For CPP_RUN, inspect the final source-bound candidate regression artifact after reconciling prerequisite build success. What exit code does it report? | `0` |
| 9 | For NEGATIVE_RUN, read its structured findings and result gaps. Which classification replaces the model's `confirmed_gap` claims? | `hypothesis` |
| 10 | For NEGATIVE_RUN, reconcile the completed status with findings and verification gaps. What outcome prevents false positives from becoming verified improvements? | `unverified` |

These questions have stable answers within retained immutable runs. They supplement the executable MCP transport suite; they are not a claim that a separate LLM evaluator has passed this corpus.
