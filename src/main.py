import numpy as np
from domain.masking import create_causal_mask
from domain.cross_attention import CrossAttention

def test_causal_mask():
    seq_len = 5
    M = create_causal_mask(seq_len)
    print("--- Tarefa 1: Máscara Causal (M) ---")
    print(M)
    print("\n(Máscara testada com sucesso no commit anterior)\n")

def test_cross_attention():
    batch_size = 1
    seq_len_frances = 10  
    seq_len_ingles = 4    
    d_model = 512
    
    np.random.seed(42)
    encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)
    decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)
    
    cross_attn = CrossAttention(d_model=d_model)
    
    output = cross_attn.forward(encoder_out=encoder_output, decoder_state=decoder_state)
    
    print("--- Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention) ---")
    print(f"Shape do Encoder Output (K, V): {encoder_output.shape}")
    print(f"Shape do Decoder State (Q): {decoder_state.shape}")
    print(f"Shape da Saída da Atenção: {output.shape}")
    
    if output.shape == (batch_size, seq_len_ingles, d_model):
        print("\nSucesso! A dimensão da saída respeita o estado do Decoder.")

def main():
    test_causal_mask()
    test_cross_attention()

if __name__ == "__main__":
    main()