import numpy as np
from domain.masking import create_causal_mask
from domain.cross_attention import CrossAttention
from domain.generator import DecoderMock

def test_causal_mask():
    seq_len = 5
    M = create_causal_mask(seq_len)
    print("--- Tarefa 1: Máscara Causal (M) ---")
    print(M)

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
    print(f"Shape da Saída da Atenção: {output.shape}\n")

def test_autoregressive_loop():
    print("--- Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo ---")
    
    encoder_out = np.random.randn(1, 10, 512)
    decoder_mock = DecoderMock(vocab_size=10000)
    
    current_sequence = ["<START>"]
    max_steps = 20
    step = 1
    
    print(f"Contexto inicial: {current_sequence}")
    
    while step <= max_steps:
        probs = decoder_mock.generate_next_token(current_sequence, encoder_out)
        
        next_token_id = int(np.argmax(probs))
        next_word = decoder_mock.id_to_word.get(next_token_id, f"token_{next_token_id}")
        
        current_sequence.append(next_word)
        print(f"Passo {step}: Previu o token '{next_word}' (ID: {next_token_id})")
        
        if next_word == "<EOS>":
            print("\n>> Parada detectada: O modelo previu o token <EOS>.")
            break
            
        step += 1

    print(f"Frase Final: {' '.join(current_sequence)}\n")

def main():
    test_causal_mask()
    test_cross_attention()
    test_autoregressive_loop()

if __name__ == "__main__":
    main()