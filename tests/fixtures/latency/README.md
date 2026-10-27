Request latency analytics
=========================

The request monitor passes lists and single-use generators of numeric samples
to `mean_latency(samples)` in latency.py. Return the arithmetic mean for any
nonempty iterable, including floats, negative samples, and a single sample.
Empty lists and exhausted generators must raise ValueError. Consume an iterable
once, with constant auxiliary memory. Keep the function name and signature.
