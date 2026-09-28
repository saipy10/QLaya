# QLaya

Multilingual, non-autoregressive System 1 decision engine: typed `choice`, `score` and `noul`
decisions over any state, in a single forward pass.

A `choice` question picks one of several options, a `score` question places the state on a
scale, and a `noul` question gives the probability that the answer is yes. QLaya runs on your own
hardware, reads 100+ languages, and its `Router` picks the checkpoint for each request.

## Start here

Install QLaya and make a first decision with the README's
[Installation](https://github.com/saipy10/QLaya#installation) and
[Quickstart](https://github.com/saipy10/QLaya#quickstart). They stay in the README so there
is one copy to keep current.

## Find a guide

| To | Read |
|---|---|
| get typed values back from a JSON schema or a pydantic model | [Schema-driven decisions](structured.md) |
| log, redact, cache or gate every decision without forking QLaya | [Prediction hooks](hooks/index.md) |
| adopt QLaya incrementally without granting execution permission | [Staged adoption](staged-adoption.md) |
| route, screen or triage inside a LangChain or LangGraph app | [LangChain & LangGraph](langchain.md) |
| run the SDK or the `qlaya-serve` HTTP API in a container, on CPU or an NVIDIA GPU | [Docker quickstart](docker.md) |
| build for ARM64 hosts or DGX Spark | [ARM64 and DGX Spark containers](docker-platforms.md) |
| specialise a checkpoint for your own decisions | [Browser-agent fine-tuning example](finetune_browser_agent.md) and the [fine-tuning notebook](https://github.com/saipy10/QLaya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) |
| judge benchmark results and deployment limits | [Benchmarks and known limits](benchmarks.md) |
| look up a class, function or parameter | [Python API reference](reference/index.md) |
| score a labelled dataset, compare to a baseline, or gate a build on it | [Evaluation harness](evals.md) |

Routing, the HTTP API, the command line, the MCP server and confidence gating are in the
[README](https://github.com/saipy10/QLaya#readme) for now.
