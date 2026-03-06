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


# ──────────────────────────────────────────────
# TAREFA 1 — Máscara Causal (Look-Ahead Mask)
# ──────────────────────────────────────────────

# isso zera os tokens futuros — triângulo superior vira -inf
def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


seq_len = 5
d_k     = 64

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

mascara           = create_causal_mask(seq_len)
scores            = Q @ K.T / np.sqrt(d_k)
scores_mascarados = scores + mascara

# aqui é onde a mágica acontece — softmax de -inf vira exatamente 0.0
pesos_atencao = softmax(scores_mascarados)

print("=" * 50)
print("TAREFA 1 — Máscara Causal")
print("=" * 50)
print("\nMatriz de pesos de atenção (arredondada):")
print(np.round(pesos_atencao, 4))
print()

# checa se todo o triângulo superior é 0.0
triangulo_sup = np.triu(pesos_atencao, k=1)
tudo_zero     = np.all(triangulo_sup == 0.0)
print(f"Triângulo superior é tudo 0.0? {tudo_zero}")
print()
