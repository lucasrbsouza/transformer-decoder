import numpy as np
from typing import List

class DecoderMock:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        
        self.id_to_word = {i: f"token_{i}" for i in range(vocab_size)}
        self.id_to_word[0] = "<START>"
        self.id_to_word[1] = "O"
        self.id_to_word[2] = "rato"
        self.id_to_word[3] = "roeu"
        self.id_to_word[4] = "<EOS>"
        
        self.step = 1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1))
        return e_x / np.sum(e_x, axis=-1)

    def generate_next_token(self, current_sequence: List[str], encoder_out: np.ndarray) -> np.ndarray:
        logits = np.random.randn(self.vocab_size)
        
        target_ids = [1, 2, 3, 4] 
        
        if self.step <= len(target_ids):
            correct_id = target_ids[self.step - 1]
            logits[correct_id] += 50.0  
            
        self.step += 1
        
        return self._softmax(logits)