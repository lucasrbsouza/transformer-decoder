import numpy as np
from domain.masking import create_causal_mask

def softmax(x: np.ndarray) -> np.ndarray:
    # Estabilização numérica subtraindo o máximo
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def main():
    seq_len = 5
    d_k = 64
    
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    M = create_causal_mask(seq_len)
    
    masked_scores = scores + M
    
    attention_weights = softmax(masked_scores)
    
    print("--- Máscara Causal (M) ---")
    print(M)
    print("\n--- Pesos de Atenção (após Softmax) ---")
    print(np.round(attention_weights, 4))

if __name__ == "__main__":
    main()