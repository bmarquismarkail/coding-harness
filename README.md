# Local LangGraph coding agent — v0.2

The v0.2 coding service adds durable jobs, isolated edits, Bubblewrap execution,
source-bound checks, SQLite LangGraph resume, and shared CLI/Python/MCP interfaces.
See **[the v0.2 guide](docs/v0.2.md)** for setup, commands, profiles, isolation,
verification rules, and live acceptance. The source checkout stays intact;
results return patches for review. A completed workflow can remain unverified.

Install the package into an active virtual environment with `python -m pip install -e .`. Configure your model endpoints and names as described below, then run a job:

```bash
python -m coding_agent.cli run \
  --wait --root /path/to/project --checks acceptance-checks.json 'Repair the bug'
```

The following sections document the preserved **legacy read-only interface**.

This repository contains a configurable coding-agent graph for OpenAI-compatible llama-server instances. It uses separate endpoints for the lead and specialist models. The default graph launches nine specialist branches in parallel, then asks the lead model to reconcile their evidence.

[agent-config.json](agent-config.json) ships with reserved `.invalid` placeholders for the lead and specialist endpoints. Set `MAIN_BASE_URL`, `MAIN_MODEL`, `SUBAGENT_BASE_URL`, and `SUBAGENT_MODEL` before making model calls, or copy the file to the Git-ignored `agent-config.local.json` and pass it with `--config`. Set `LLAMA_API_KEY` when the llama-server instances require authentication; the adapter sends it as a Bearer token. Completed conclusions are also sent to a separate, read-only independent reviewer model. Its default endpoint is OpenAI's `https://api.openai.com/v1`, using `gpt-4.1-mini` and `OPENAI_API_KEY`; override it with `REVIEWER_BASE_URL`, `REVIEWER_MODEL`, and `REVIEWER_API_KEY_ENV` or the `reviewer` config record. Main, specialist, and reviewer endpoints must be distinct. The URL must include `http://` or `https://`; the adapter appends `/chat/completions` to each configured base URL.

## Run

With your virtual environment active:

```bash
python -m coding_agent.cli --print-config
python -m coding_agent.cli \
  --root /path/to/repository \
  "Implement the requested feature and give me a verified change plan"
```

`MAIN_BASE_URL`, `MAIN_MODEL`, `SUBAGENT_BASE_URL`, `SUBAGENT_MODEL`, and `SUBAGENT_PARALLELISM` are explicit non-secret overrides. The shipped value for `SUBAGENT_PARALLELISM` is 9. The specialist roles are intentionally broad so the same graph can inspect any text-based programming language while giving Python and C/C++ dedicated attention.

Each model record also accepts `reasoning_effort` and `enable_thinking`. The
specialist record uses `"reasoning_effort": "none"` and `"enable_thinking": false`
to limit reasoning output. The adapter passes `enable_thinking` inside
`chat_template_kwargs`. The main record leaves both options unset and uses the
server template defaults. These are
[llama-server request options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Output that reaches `max_tokens` is reported as an error.

## Architecture

The graph is:

```text
context -> lead plan -> Send(specialist x 9) -> lead synthesis
```

Repository context is bounded and read-only. It includes important project files and common source/configuration extensions while excluding VCS metadata, virtual environments, build outputs, caches, symlinks, and hidden directories. Model responses are parsed as JSON when possible, but malformed responses are retained as text instead of being treated as successful structured output.

The legacy interface remains a planning/evidence handoff agent. It does not execute project commands or edit files. The v0.2 service provides editing and execution through a separate validated tool broker and isolated workspaces.

## zvec-grep tool

The package includes a typed `zvec_grep` LangChain tool backed by the installed `zg` CLI. It supports three explicit actions: `index`, `query`, and `status`. Indexing is never triggered implicitly by a query, and queries always use `--refresh off`, so a missing index cannot silently start a rebuild.

Build an index explicitly, using `ZVEC_GREP_EMBEDDING` or an explicit embedding model:

```bash
python -m coding_agent.cli \
  --root /path/to/repository \
  --zvec-index \
  --zvec-embedding local/potion-code-16m-v2
```

Query the existing index directly:

```bash
python -m coding_agent.cli \
  --root /path/to/repository \
  --zvec-query "where is the graph constructed?" \
  --zvec-route hybrid
```

To use zvec results during an agent run, provide `--zvec-query` together with the coding task. Results are navigation hints; the lead and specialist prompts are instructed to verify them against source context. The Python API is also available through `ZvecGrepTool.index()`, `.query()`, `.status()`, and `.as_langchain_tool()`.

Run the simple real-CLI smoke test with:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  -m unittest tests/test_zvec_smoke.py -v
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Ordinary tests use fake models and `httpx.MockTransport`; the zvec smoke test
also invokes the installed `zg` executable. Live model inference is opt-in:

```bash
RUN_LIVE_MODEL_TESTS=1 PYTHONDONTWRITEBYTECODE=1 \
  python \
  -m unittest discover -s tests -p test_live_models.py -v
```

This test requires `LLAMA_API_KEY` and configured model endpoints and names. It first verifies JSON responses from both
configured llama-server endpoints, then runs the complete lead → nine
specialists → synthesis graph against a small request-latency analytics project.
The task is to repair a function that fails with single-use generators and empty
iterables. The test verifies the baseline failure, extracts the model's proposed
replacement, checks its allowed Python syntax, and grades it against nine
independent cases in a child process with restricted builtins, CPU, memory, and
time limits. The test driver applies the proposal only to its own fixture;
the graph itself still produces a handoff.

Each run retains the fixture, raw responses, HTTP status, finish reasons,
request timings, graph state, and `report.json` under
`development-evidence/live-model-smoke-*`. The test does not create a search
index or execute model-proposed shell commands. Normal test discovery skips it
unless `RUN_LIVE_MODEL_TESTS=1` is set.
