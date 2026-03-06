import numpy as np

class CrossAttention:
    def __init__(self, d_model: int):
        if d_model <= 0:
            raise ValueError("d_model deve ser maior que zero.")
            
        self.d_model = d_model
        
        np.random.seed(42)
        self.w_q = np.random.randn(d_model, d_model) * 0.01
        self.w_k = np.random.randn(d_model, d_model) * 0.01
        self.w_v = np.random.randn(d_model, d_model) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, encoder_out: np.ndarray, decoder_state: np.ndarray) -> np.ndarray:
        # 1. Projeção: O estado do Decoder vira a Query (Q)
        q = decoder_state @ self.w_q
        
        # 2. Projeção: A memória do Encoder vira Key (K) e Value (V)
        k = encoder_out @ self.w_k
        v = encoder_out @ self.w_v

        # 3. Cálculo da Atenção (Sem máscara causal)
        k_t = np.swapaxes(k, -2, -1)
        
        scores = (q @ k_t) / np.sqrt(self.d_model)
        attention_weights = self._softmax(scores)
        
        return attention_weights @ v