# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/SMASH-Mash/prompt-improver-mcp
cd prompt-improver-mcp
uv sync --group dev
uv run pytest
```

## Project layout

| File | What it holds |
|---|---|
| `src/prompt_improver/techniques.py` | The technique catalog: each technique, its papers, when it applies, and how to apply it |
| `src/prompt_improver/analyzer.py` | Task profiling and the rules that select techniques |
| `src/prompt_improver/distill.py` | Offline context selection (a BM25 approximation of System 2 Attention) |
| `src/prompt_improver/compose.py` | The rewriter's meta-prompt and the host-mode template draft |
| `src/prompt_improver/llm.py` | The Claude API backend |
| `src/prompt_improver/engine.py` | The pipeline that ties these together |
| `src/prompt_improver/server.py` | The MCP server: tools, the `improve` prompt, MCP sampling |
| `src/prompt_improver/cli.py` | The `prompt-improver` command |

## Adding a technique

1. Add a `Paper` and a `Technique` to `techniques.py`. Check the arXiv ID against the paper title; every citation in
   this project has been verified, so keep it that way.
2. If the technique changes how a prompt is phrased for models with built-in thinking, set `how_reasoning`.
3. Add a selection rule in `select_techniques` (`analyzer.py`), and a scaffold in `compose.py` if host-mode drafts
   should apply it.
4. Add a test, and a row to the technique table in the README.

## Pull requests

- Keep changes focused, and run `uv run pytest` before opening the PR.
- If you change the meta-prompt or the selection rules, include an example input with the output before and after
  the change.
