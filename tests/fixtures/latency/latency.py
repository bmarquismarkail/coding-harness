"""Latency statistics used by a request-monitoring application."""


def mean_latency(samples):
    """Return the arithmetic mean; raise ValueError for an empty iterable."""
    if not samples:
        raise ValueError("no latency samples")
    return sum(samples) / len(samples)
