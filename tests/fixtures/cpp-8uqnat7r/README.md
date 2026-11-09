# Historical false-rejection evidence

The artifacts in this directory are exact bytes copied from the retained run identified by `record.json`. Every artifact filename is its SHA-256. Baseline and candidate source copies preserve the original bytes, including the candidate's missing final newline.

Before any production edits, `/tmp/replay_harness_baseline.py` replayed the original check receipts and specialist findings/gaps through `Checks.finish` at commit `86a75251b1ad64a698926ac623d1dcc58d5be680`. `baseline-result.json` records the resulting unverified outcome, exact remaining gap, and original candidate fingerprint. No check was executed or inferred during this replay. The original retained run was not changed.

`tests/test_v3.py` verifies this baseline behavior and explicitly projects the archived specialist observation and missing-check assertions into the v0.3 reviewer contract for the revised acceptance pair. The original report remains attached to the proposal. This exercises the failure mechanism with original receipts, rather than asserting a newly inferred historical model result.
