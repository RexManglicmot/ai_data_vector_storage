import numpy as np
from app.lib_utils import l2norm, percentile, bench_latency, to_dtype_and_norm, recall_at_k_from_labels

X = np.array([[3.0,4.0],[0.0,5.0]], dtype=np.float32)
Xn = l2norm(X)
assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0, atol=1e-6)

arr = np.array([0,1,2,3,4,5], dtype=np.float32)
assert percentile(arr, 50) == 2.5

M = to_dtype_and_norm(np.random.randn(100, 16).astype(np.float32), np.float32)
Q = to_dtype_and_norm(np.random.randn(5,   16).astype(np.float32), np.float32)
p50, p95 = bench_latency(M, Q, k=10)
assert p50 >= 0 and p95 >= 0

# recall util
topk = [np.array([0,1,2]), np.array([3,4,5])]
q_labels = ["a","b"]
corpus_labels = ["a","x","x","y","b","z"]
r = recall_at_k_from_labels(topk, q_labels, corpus_labels, k=3)
assert 0.0 <= r <= 1.0
print("✅ lib_utils tests OK")

# Run 