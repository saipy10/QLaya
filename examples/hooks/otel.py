"""Emit a counter and a latency histogram for every decision.

Works without any dependency; uncomment the OpenTelemetry lines if it is installed.

    python examples/hooks/otel.py
"""
import qlaya

COUNTS = {}
LATENCIES = []


def on_predict_end(ctx):
    model = ctx.model or "auto"
    COUNTS[model] = COUNTS.get(model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)
    # With OpenTelemetry installed (`pip install opentelemetry-api`):
    #   from opentelemetry import metrics
    #   meter = metrics.get_meter("qlaya")
    #   meter.create_counter("qlaya.decisions").add(1, {"model": model})
    #   meter.create_histogram("qlaya.decision.latency_ms").record(ctx.elapsed_ms or 0.0)


agent = qlaya.load("saipy10/qlaya", on_predict_end=on_predict_end)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
print("counts:", COUNTS)
print("latency samples:", len(LATENCIES))
