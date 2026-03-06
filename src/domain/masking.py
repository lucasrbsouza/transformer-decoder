import numpy as np

def create_causal_mask(seq_len: int) -> np.ndarray:
    # Cria uma matriz com True acima da diagonal principal
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    
    # Inicializa com zeros (triângulo inferior e diagonal)
    causal_mask = np.zeros((seq_len, seq_len))
    
    # Preenche o triângulo superior com infinito negativo
    causal_mask[mask] = -np.inf
    
    return causal_mask