# Código

```
code/
  analise/    analise_resultados.py — gera todas as tabelas de resultado do artigo
  execucao/   a pipeline que produziu os resultados, em cópia congelada
```

## `analise/`

`analise_resultados.py` lê os arquivos de `results/` e reproduz cada tabela
reportada no artigo, incluindo as contagens por direção, a eficiência por
célula e as verificações de consistência. Depende apenas de `pandas`,
`openpyxl` e `scipy`.

```bash
pip install pandas pyarrow openpyxl scipy
python code/analise/analise_resultados.py
```

## `execucao/`

Preparação dos dados, recuperação densa, laço de entrega progressiva com
parada declarada pelo gerador e sonda de entropia de *logits* — na ordem dos
prefixos numéricos:

| Arquivo | O que faz |
|---|---|
| `00_preparar_dados.py` | segmenta e indexa o corpus; valida o mapeamento gabarito ↔ base |
| `01_embeddings_perguntas.py` | calcula e congela em disco os vetores das consultas |
| `02_bloco1_desempenho.py` | Bloco 1: os três modos de entrega e as duas modalidades de parada |
| `03_bloco2_entropia.py` | Bloco 2: a sonda de incerteza, com registro de *logits* |
| `04_metricas.py` | consolida as execuções e calcula as métricas dos três eixos |
| `05_piloto.py` | execução reduzida, para verificar a pilha antes de uma rodada longa |
| `06_diagnostico_retriever.py` | diagnóstico do recuperador e da convenção de codificação |
| `comum.py` | montagem de mensagens, critérios de parada, contabilidade de *tokens* |
| `verificar_ambiente.py`, `verificar_memoria.py` | conferências de pilha e de memória antes de rodar |

O ambiente de execução está no `Dockerfile` e no `docker-compose.yml`, com as
versões fixadas em `requirements.lock`. `DOCKER.md` descreve o uso.
`.env.exemplo` lista as variáveis esperadas — o `.env` real, com o token de
acesso a repositórios de pesos, nunca é distribuído.

### Leia isto antes de auditar um número

**[`execucao/DECISOES_TECNICAS.md`](execucao/DECISOES_TECNICAS.md)** registra as
decisões de implementação que afetam os valores reportados, com o motivo de
cada uma e a coluna por onde ela pode ser conferida: a regra de decisão sob o
contrato JSON e a alternativa calculada retroativamente; o alinhamento da
janela de entropia; por que a medida principal da sonda é o posto e não a taxa
de acerto; por que o comparador de similaridade é uma identidade e não um
baseline; o tratamento de censura e do teto de *tokens*; e as condições
implementadas mas não usadas no artigo.

Várias dessas escolhas não são neutras — mudam o valor de $D$, o que entra nas
médias de entropia ou o que conta como parada válida. Quem for conferir um
número precisa saber qual regra estava em vigor.

### Esta é uma cópia congelada

O código foi desenvolvido pela equipe em um repositório de trabalho privado.
O que está aqui é a **versão final usada nas execuções reportadas**, copiada
arquivo a arquivo: o histórico de *commits*, os ramos e as pastas de dados e
de saídas não acompanham a cópia.

As execuções reportadas correspondem a dois pontos desse histórico —
`c196c1b` para o braço de recuperação limitada, empacotado em 10/09/2026, e
`d26caf0` para o braço de recuperação competente, empacotado em 13/09/2026.
Os manifestos em `results/env/` trazem os digestos SHA-256 de cada arquivo de
resultado produzido por eles, o que permite amarrar um número ao código e à
pilha de software que o geraram.

O código é obra dos cinco autores do artigo, creditados em `CITATION.cff` e no
cabeçalho da licença.

### Reproduzir sem o corpus

`02_*` e `03_*` precisam do texto dos documentos normativos, que não é
distribuído (ver [Sobre o que não está aqui](../README.md#sobre-o-que-não-está-aqui)).
Eles estão aqui para leitura e auditoria — para conferir como o laço de parada
decide, como os *prompts* são montados e como os *tokens* são contados —, não
para reexecução direta.

O que **é** reexecutável a partir deste repositório é `04_metricas.py` e todo o
`analise/`, que operam sobre os arquivos de resultado já liberados. É por eles
que passa a reprodução de cada número das Seções 6 a 8 do artigo.

Os *prompts* integrais, o conjunto de padrões do autômato de parada e o
registro de ambiente de cada execução estão no Online Resource 1 do artigo.
