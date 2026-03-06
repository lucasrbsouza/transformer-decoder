# Transformer Decoder "From Scratch" - Laboratório 3

**Instituição:** iCEV - Instituto de Ensino Superior  
**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Aluno:** Lucas Souza

## 1. Objetivo do Laboratório

Este repositório contém a implementação dos blocos matemáticos centrais do Decoder de um modelo Transformer. O sistema foi construído para gerar texto de forma fluente, garantindo o mascaramento adequado para impedir que o modelo "olhe para o futuro" durante o processamento.

O laboratório engloba três pilares fundamentais:
1. **Máscara Causal (Look-Ahead Masking):** Implementação da álgebra linear que zera as probabilidades de tokens futuros.
2. **Cross-Attention (Ponte Encoder-Decoder):** Integração estrutural onde o estado gerado pelo Decoder atua como *Query* sobre a memória (*Keys* e *Values*) do Encoder.
3. **Loop de Inferência Auto-Regressivo:** Simulação da geração de texto palavra por palavra, interrompida condicionalmente ao detectar o token `<EOS>`.

O projeto foi desenvolvido inteiramente com `Python 3.11`, `numpy` e `pandas`, sem o uso de bibliotecas prontas de Deep Learning.

## 2. Arquitetura do Projeto

O código foi estruturado seguindo princípios de **Clean Architecture**, **SOLID** e **Domain-Driven Design (DDD)**, isolando regras matemáticas puras do fluxo de execução:

- `src/domain/masking.py`: Lógica de criação da matriz triangular para a máscara causal.
- `src/domain/cross_attention.py`: Componente de atenção assimétrica (sem máscara) cruzando fronteiras de tensores.
- `src/domain/generator.py`: Mock do Decoder contendo a projeção final para o vocabulário e função *Softmax*.
- `src/main.py`: Maestro principal que executa e valida matematicamente cada uma das três tarefas exigidas.

## 3. Requisitos

- Python 3.8+
- NumPy
- Pandas

## 4. Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/lucasrbsouza/transformer-decoder.git
   cd transformer_decoder
   ```

2. Instale as dependências:
   ```bash
   pip install numpy pandas
   ```

3. Execute o script de validação:
   ```bash
   python src/main.py
   ```

### Saída Esperada

O console demonstrará a validação progressiva das três tarefas: a matriz de máscara com infinito negativo, a conservação dimensional na ponte *Cross-Attention*, e o loop auto-regressivo gerando a frase até a condição de parada:

```text
--- Tarefa 1: Máscara Causal (M) ---
[[  0. -inf -inf -inf -inf]
 [  0.   0. -inf -inf -inf]
 [  0.   0.   0. -inf -inf]
 [  0.   0.   0.   0. -inf]
 [  0.   0.   0.   0.   0.]]

--- Tarefa 2: A Ponte Encoder-Decoder (Cross-Attention) ---
Shape do Encoder Output (K, V): (1, 10, 512)
Shape do Decoder State (Q): (1, 4, 512)
Shape da Saída da Atenção: (1, 4, 512)

--- Tarefa 3: Simulando o Loop de Inferência Auto-Regressivo ---
Contexto inicial: ['<START>']
Passo 1: Previu o token 'O' (ID: 1)
Passo 2: Previu o token 'rato' (ID: 2)
Passo 3: Previu o token 'roeu' (ID: 3)
Passo 4: Previu o token '<EOS>' (ID: 4)

>> Parada detectada: O modelo previu o token <EOS>.
Frase Final: <START> O rato roeu <EOS>
```

## 5. Nota de Integridade e Créditos

Em conformidade com as exigências pedagógicas, este projeto foi construído traduzindo as equações e conceitos lecionados para o código. Ferramentas de Inteligência Artificial foram utilizadas estritamente para suporte na refatoração arquitetural, tirar duvidas sobre implementação de regras, e suporte a Documentação do projeto, mantendo a integridade da lógica matemática original.