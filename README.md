# Local LangGraph coding agent

This repository contains a configurable coding-agent graph for OpenAI-compatible llama-server instances. It uses one lead model at `https://lead.example.invalid/v1` and a separate subagent model at `https://specialist.example.invalid/v1`. The default graph launches nine specialist branches in parallel, then asks the lead model to reconcile their evidence.

The default configuration is in [agent-config.json](agent-config.json). It contains model names and endpoint locations, but no credential. Set `LLAMA_API_KEY` when the servers require authentication; the adapter sends it as a Bearer token. The URL must include `http://` or `https://`; the adapter appends `/chat/completions` to each configured base URL. If the reverse proxy is plain HTTP, set `MAIN_BASE_URL=http://lead.example.invalid/v1` and `SUBAGENT_BASE_URL=http://specialist.example.invalid/v1` (or edit the JSON file).

## Run

Use the requested environment:

```bash
/home/user/python-venv/langchain-suite/bin/python -m coding_agent.cli --print-config
/home/user/python-venv/langchain-suite/bin/python -m coding_agent.cli \
  --root /path/to/repository \
  "Implement the requested feature and give me a verified change plan"
```

`MAIN_BASE_URL`, `MAIN_MODEL`, `SUBAGENT_BASE_URL`, `SUBAGENT_MODEL`, and `SUBAGENT_PARALLELISM` are explicit non-secret overrides. The shipped value for `SUBAGENT_PARALLELISM` is 9. The specialist roles are intentionally broad so the same graph can inspect any text-based programming language while giving Python and C/C++ dedicated attention.

Each model record also accepts `reasoning_effort` and `enable_thinking`. The
specialist record uses `"reasoning_effort": "none"` and `"enable_thinking": false`
so Spark returns its report within the configured token budget. Its installed
template requires the explicit `enable_thinking` switch; `reasoning_effort` alone
did not suppress thinking. The adapter passes `enable_thinking` inside
`chat_template_kwargs`. The main record omits both settings, preserving Mellum's
default thinking behavior. These are
[llama-server request options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Output that reaches `max_tokens` is reported as an error.

## Architecture

The graph is:

```text
context -> lead plan -> Send(specialist x 9) -> lead synthesis
```

Repository context is bounded and read-only. It includes important project files and common source/configuration extensions while excluding VCS metadata, virtual environments, build outputs, caches, symlinks, and hidden directories. Model responses are parsed as JSON when possible, but malformed responses are retained as text instead of being treated as successful structured output.

The current implementation is intentionally a planning/evidence handoff agent. It does not execute arbitrary model-generated shell commands or edit files. That keeps the provider boundary safe and makes the graph testable without network access. A later tool broker can add typed, allow-listed operations such as exact search, read-file ranges, compiler checks, and sandboxed tests.

## zvec-grep tool

The package includes a typed `zvec_grep` LangChain tool backed by the installed `zg` CLI. It supports three explicit actions: `index`, `query`, and `status`. Indexing is never triggered implicitly by a query, and queries always use `--refresh off`, so a missing index cannot silently start a rebuild.

Build an index explicitly, using `ZVEC_GREP_EMBEDDING` or an explicit embedding model:

```bash
/home/user/python-venv/langchain-suite/bin/python -m coding_agent.cli \
  --root /path/to/repository \
  --zvec-index \
  --zvec-embedding local/potion-code-16m-v2
```

Query the existing index directly:

```bash
/home/user/python-venv/langchain-suite/bin/python -m coding_agent.cli \
  --root /path/to/repository \
  --zvec-query "where is the graph constructed?" \
  --zvec-route hybrid
```

To use zvec results during an agent run, provide `--zvec-query` together with the coding task. Results are navigation hints; the lead and specialist prompts are instructed to verify them against source context. The Python API is also available through `ZvecGrepTool.index()`, `.query()`, `.status()`, and `.as_langchain_tool()`.

Run the simple real-CLI smoke test with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/python-venv/langchain-suite/bin/python \
  -m unittest tests/test_zvec_smoke.py -v
```

## Tests

```bash
/home/user/python-venv/langchain-suite/bin/python -m unittest discover -s tests -v
```

Ordinary tests use fake models and `httpx.MockTransport`; the zvec smoke test
also invokes the installed `zg` executable. Live model inference is opt-in:

```bash
RUN_LIVE_MODEL_TESTS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/user/python-venv/langchain-suite/bin/python \
  -m unittest discover -s tests -p test_live_models.py -v
```

This test requires `LLAMA_API_KEY`. It first verifies JSON responses from both
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
