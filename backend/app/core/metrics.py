"""
FiniRAG metrics — a small, dependency-free Prometheus-compatible registry.

Why dependency-free: the deployment target is a single small web instance, and
these metrics are a first-class part of the app rather than an add-on. The
registry below implements exactly what a scrape needs — counters, gauges and
histograms with labels — and renders the Prometheus text exposition format, so
any standard scraper (Prometheus, Grafana Agent, Render metrics) can consume
it without an SDK.

What production RAG actually needs to see, and what is recorded here:

  RED per route            request rate, error rate, latency histogram
  Retrieval health         path used (pgvector/fts/fallback), degradation,
                           candidate counts, empty-result rate, latency by stage
  Embedding health         call count by outcome, batch size, latency
  Generation health        answer provider mix, fallback reasons, abstention
                           rate, latency, and token usage (cost driver)
  Grounding                citation repairs, claims dropped, unsupported rate
  Indexing pipeline        jobs by terminal state, chunks written, latency,
                           pending backlog gauge

Labels are deliberately low-cardinality (enumerated values, route templates —
never raw ids, filenames or query text). No secrets, document text or amounts
are ever recorded.
"""

from __future__ import annotations

import threading
from collections import defaultdict

# Latency buckets in seconds, spanning fast DB ops to slow LLM calls. These are
# the histogram boundaries reported per observation.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                   2.5, 5.0, 10.0, 30.0, 60.0)
# Smaller buckets for counts (candidates, batch sizes).
COUNT_BUCKETS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0)


def _labels_key(labels: dict | None) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _render_labels(labels: dict | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class _Counter:
    def __init__(self):
        self.values: dict[tuple, float] = defaultdict(float)

    def inc(self, labels: dict | None, amount: float = 1.0) -> None:
        self.values[_labels_key(labels)] += amount


class _Gauge:
    def __init__(self):
        self.values: dict[tuple, float] = {}

    def set(self, labels: dict | None, value: float) -> None:
        self.values[_labels_key(labels)] = float(value)

    def inc(self, labels: dict | None, amount: float = 1.0) -> None:
        key = _labels_key(labels)
        self.values[key] = self.values.get(key, 0.0) + float(amount)

    def dec(self, labels: dict | None, amount: float = 1.0) -> None:
        self.inc(labels, -amount)


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts: dict[tuple, list[int]] = {}
        self.sums: dict[tuple, float] = defaultdict(float)
        self.totals: dict[tuple, int] = defaultdict(int)

    def observe(self, labels: dict | None, value: float) -> None:
        key = _labels_key(labels)
        if key not in self.counts:
            self.counts[key] = [0] * (len(self.buckets) + 1)
        value = float(value)
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[key][i] += 1
        self.counts[key][-1] += 1  # +Inf
        self.sums[key] += value
        self.totals[key] += 1


class MetricsRegistry:
    """Thread-safe in-process registry. Single instance: `metrics`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}

    # ── registration ──

    def counter(self, name: str):
        with self._lock:
            return self._counters.setdefault(name, _Counter())

    def gauge(self, name: str):
        with self._lock:
            return self._gauges.setdefault(name, _Gauge())

    def histogram(self, name: str, buckets: tuple[float, ...] = LATENCY_BUCKETS):
        with self._lock:
            return self._histograms.setdefault(name, _Histogram(buckets))

    # ── convenience recorders ──

    def inc(self, name: str, labels: dict | None = None, amount: float = 1.0):
        self.counter(name).inc(labels, amount)

    def set_gauge(self, name: str, value: float, labels: dict | None = None):
        self.gauge(name).set(labels, value)

    def observe(self, name: str, value: float, labels: dict | None = None,
                buckets: tuple[float, ...] = LATENCY_BUCKETS):
        self.histogram(name, buckets).observe(labels, value)

    # ── export ──

    def snapshot(self) -> dict:
        """Plain-dict view for tests and the JSON variant of /metrics."""
        with self._lock:
            return {
                "counters": {n: {str(k): v for k, v in c.values.items()}
                             for n, c in self._counters.items()},
                "gauges": {n: {str(k): v for k, v in g.values.items()}
                           for n, g in self._gauges.items()},
                "histograms": {
                    n: {"count": sum(h.totals.values()),
                        "sum": round(sum(h.sums.values()), 6),
                        "series": len(h.totals)}
                    for n, h in self._histograms.items()},
            }

    def render_prometheus(self) -> str:
        """Prometheus text exposition format (v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            for name, counter in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                if not counter.values:
                    lines.append(f"{name} 0")
                for key, value in sorted(counter.values.items()):
                    lines.append(f"{name}{_render_key(key)} {_fmt(value)}")
            for name, gauge in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                if not gauge.values:
                    lines.append(f"{name} 0")
                for key, value in sorted(gauge.values.items()):
                    lines.append(f"{name}{_render_key(key)} {_fmt(value)}")
            for name, hist in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                for key in sorted(hist.counts):
                    cumulative = 0
                    for i, bound in enumerate(hist.buckets):
                        cumulative = hist.counts[key][i]
                        lines.append(
                            f"{name}_bucket{_render_key(key, extra=('le', _fmt(bound)))} "
                            f"{cumulative}")
                    lines.append(
                        f"{name}_bucket{_render_key(key, extra=('le', '+Inf'))} "
                        f"{hist.counts[key][-1]}")
                    lines.append(f"{name}_sum{_render_key(key)} {_fmt(hist.sums[key])}")
                    lines.append(f"{name}_count{_render_key(key)} {hist.counts[key][-1]}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Test/verification helper: clear all series."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


def _render_key(key: tuple, extra: tuple | None = None) -> str:
    pairs = {k: v for k, v in key}
    if extra:
        pairs[extra[0]] = extra[1]
    if not pairs:
        return ""
    return _render_labels(pairs)


def _fmt(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


# ── Metric names (single source of truth; avoid string drift) ──

HTTP_REQUESTS = "finirag_http_requests_total"
HTTP_LATENCY = "finirag_http_request_duration_seconds"

RETRIEVAL_REQUESTS = "finirag_retrieval_requests_total"
RETRIEVAL_LATENCY = "finirag_retrieval_duration_seconds"
RETRIEVAL_CANDIDATES = "finirag_retrieval_candidates"
RETRIEVAL_EMPTY = "finirag_retrieval_empty_total"

EMBEDDING_REQUESTS = "finirag_embedding_requests_total"
EMBEDDING_LATENCY = "finirag_embedding_duration_seconds"
EMBEDDING_BATCH = "finirag_embedding_batch_size"

ANSWER_REQUESTS = "finirag_answer_requests_total"
ANSWER_LATENCY = "finirag_answer_duration_seconds"
ANSWER_FALLBACKS = "finirag_answer_fallbacks_total"
LLM_TOKENS = "finirag_llm_tokens_total"
CLAIMS_DROPPED = "finirag_claims_dropped_total"
CLAIMS_REPAIRED = "finirag_claims_repaired_total"
CITATIONS_EMITTED = "finirag_citations_emitted_total"

INDEXING_JOBS = "finirag_indexing_jobs_total"
INDEXING_CHUNKS = "finirag_indexing_chunks_total"
INDEXING_LATENCY = "finirag_indexing_duration_seconds"
INDEXING_PENDING = "finirag_indexing_pending_jobs"

DOCUMENTS_PROCESSED = "finirag_documents_processed_total"
DOCUMENT_LATENCY = "finirag_document_processing_duration_seconds"

# The process-wide registry.
metrics = MetricsRegistry()
