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


# ──────────────────────────────────────────────
# TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)
# ──────────────────────────────────────────────

# encoder gerou contexto da frase de origem (ex: francês)
encoder_output = np.random.randn(1, 10, D_MODEL)

# decoder já gerou alguns tokens da frase de destino (ex: inglês)
decoder_state  = np.random.randn(1, 4, D_MODEL)

# decoder pergunta (Q), encoder responde com K e V — sem máscara aqui
def cross_attention(encoder_out, dec_state):
    Wq = np.random.randn(D_MODEL, D_MODEL)
    Wk = np.random.randn(D_MODEL, D_MODEL)
    Wv = np.random.randn(D_MODEL, D_MODEL)

    Q = dec_state   @ Wq   # [1, 4, 512]
    K = encoder_out @ Wk   # [1, 10, 512]
    V = encoder_out @ Wv   # [1, 10, 512]

    d_k    = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # [1, 4, 10]
    pesos  = softmax(scores)
    return pesos @ V   # [1, 4, 512]


contexto = cross_attention(encoder_output, decoder_state)

print("=" * 50)
print("TAREFA 2 — Cross-Attention")
print("=" * 50)
print(f"\nencoder_output shape : {encoder_output.shape}")
print(f"decoder_state shape  : {decoder_state.shape}")
print(f"contexto shape       : {contexto.shape}")
print()


# ──────────────────────────────────────────────
# TAREFA 3 — Loop Auto-Regressivo
# ──────────────────────────────────────────────

# vocabulário fictício — índice 0 é <START>, índice 1 é <EOS>
vocab = ["<START>", "<EOS>"] + [f"word_{i}" for i in range(2, VOCAB_SIZE)]

# simula a passagem pelo decoder — retorna distribuição sobre o vocabulário
def generate_next_token(sequencia_atual, encoder_out):
    logits = np.random.randn(VOCAB_SIZE)

    # força o <EOS> depois de 5 tokens gerados
    if len(sequencia_atual) >= 5:
        logits[1] = 100.0

    return softmax(logits)


# loop de inferência — uma palavra por vez até aparecer o <EOS>
sequencia = ["<START>"]

print("=" * 50)
print("TAREFA 3 — Loop Auto-Regressivo")
print("=" * 50)
print()

passo = 0
while True:
    probs         = generate_next_token(sequencia, encoder_output)
    proximo_id    = np.argmax(probs)
    proximo_token = vocab[proximo_id]

    sequencia.append(proximo_token)
    passo += 1
    print(f"Passo {passo}: token gerado → '{proximo_token}'")

    if proximo_token == "<EOS>":
        break

print()
print(f"Frase gerada: {' '.join(sequencia)}")
