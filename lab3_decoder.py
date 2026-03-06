import numpy as np

# Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k) + M) V

D_MODEL    = 512
VOCAB_SIZE = 10_000

np.random.seed(42)

# softmax manual — subtrai o max pra não explodir com -inf
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)
