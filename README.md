# PROBE — construção incremental de contexto em RAG local sobre documentação normativa hospitalar (pt-BR)

Dados, resultados e código de análise do artigo:

> da Cunha, M. V.; Silveira, M. R.; Sperb, C. B.; de Freitas, L. A.; Corrêa, U. B.
> **PROBE: Incremental Context Construction with Generator-Declared Stopping in
> Local Retrieval-Augmented Generation over Portuguese Hospital Documents.**
> *Language Resources and Evaluation*, submetido.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22858423.svg)](https://doi.org/10.5281/zenodo.22858423)
[![Código: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Dados: CC BY 4.0](https://img.shields.io/badge/dados-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

---

## O que é isto

O PROBE entrega os segmentos recuperados **progressivamente** ao longo de um
*ranking* fixo, e o próprio gerador declara quando o contexto é suficiente.
Este repositório contém o recurso de avaliação em português, os resultados
por consulta dos dois blocos experimentais e o código de análise que produz
todas as tabelas do artigo.

**O que se pode reproduzir integralmente a partir daqui:** as análises de
profundidade de parada, de custo em *tokens*, de fidelidade semântica e a
sonda de incerteza — isto é, todos os números reportados nas Seções 6 a 8.

**O que não se pode:** reexecutar a recuperação ou a geração sobre o texto
original. O código que faz isso está em `code/execucao/`, para leitura e
auditoria, mas o texto integral dos documentos normativos não é liberado (ver
[Sobre o que não está aqui](#sobre-o-que-não-está-aqui)).

---

## O recurso de avaliação

| | |
|---|---|
| Documentos normativos | 107, de 5 setores de um hospital universitário |
| Segmentos indexados | 740, apontando para 462 páginas distintas |
| Pares pergunta–resposta | **192** |
| &nbsp;&nbsp;subconjunto sintético, com segmento-ouro rastreável | 74 |
| &nbsp;&nbsp;subconjunto de especialistas | 118, formulados por 12 profissionais |
| Questões com mais de um segmento-ouro | 12 |
| Língua | português brasileiro |
| Gênero textual | documentação normativa institucional (POPs, manuais, regulamentos) |

Até onde sabemos, é o único conjunto público de documentação normativa
hospitalar **institucional** em português anotado para avaliação de
recuperação e geração. Os corpora clínicos disponíveis em português são de
narrativa e prontuário; o recurso aberto de nível ministerial existente cobre
outro nível de autoridade e outro versionamento.

⚠️ **Os 192 pares estão ordenados por origem no arquivo.** Uma amostra tomada
pelas *n* primeiras questões mede exclusivamente o subconjunto sintético, que
é sistematicamente mais fácil. Para amostrar, espace igualmente sobre
`pergunta_idx`.

---

## O desenho experimental

```
192 consultas × 3 geradores × 2 modalidades de parada × 3 modos de entrega × 2 recuperadores
= 6 912 execuções   →   2 304 comparações pareadas em 12 células de N = 192
```

| Fator | Níveis |
|---|---|
| Geradores | gpt-oss:20b · Mistral-Small-3.1-24B · Qwen3.8:27B (todos locais) |
| Recuperadores | Qwen3-Embedding-8B (`qwen3`) · multilingual-e5-small (`e5small`) |
| Modalidades de parada | A: autômato de expressões regulares · B: contrato de saída JSON |
| Modos de entrega | iterativo (substitui) · incremental (acumula) · saturante (bloco único, *k* = 50) |

**Bloco 2 — sonda de incerteza:** 74 consultas × 10 posições embaralhadas ×
3 geradores × 2 modalidades × 2 recuperadores = 888 consultas-condição e
**8 880 inferências com registro de *logits***.

---

## Estrutura

```
data/
  perguntas_192.parquet            as 192 consultas e respostas de referência
  perguntas_74_gold.parquet        subconjunto com segmento-ouro rastreável
  corpus_index_e5small.parquet     740 segmentos: ids, vetores densos e comprimentos
  corpus_index_qwen3.parquet       — SEM o texto dos documentos
  embeddings_perguntas_*.parquet   vetores densos das consultas, por recuperador
  README.md                        dicionário de dados, coluna a coluna
  CHECKSUMS.sha256

results/
  bloco1/                          laço de parada: consolidado por recuperador e tabelas
  bloco2/                          sonda de entropia: postos, nulas, circularidade
  analise/                         auditoria consulta a consulta e planilha consolidada
  env/                             metadados de ambiente e manifestos das entregas
  README.md                        dicionário das colunas, arquivo por arquivo
  CHECKSUMS.sha256

code/
  analise/                         gera todas as tabelas de resultado do artigo
  execucao/                        a pipeline dos experimentos, em cópia congelada
  execucao/DECISOES_TECNICAS.md    as escolhas de implementação que afetam os números
  README.md                        o que cada script faz, e o que é reexecutável

docs/
  notas_tecnicas/                  relatórios dos dois blocos
```

Para conferir um número do artigo são dois arquivos:
[`results/README.md`](results/README.md) diz em qual coluna de qual arquivo ele
está, e
[`code/execucao/DECISOES_TECNICAS.md`](code/execucao/DECISOES_TECNICAS.md) diz
sob qual regra ele foi produzido — várias dessas regras não são neutras.

---

## Uso rápido

```bash
git clone https://github.com/Murilovcunha/probe-rag-ptbr.git
cd probe-rag-ptbr
pip install pandas pyarrow openpyxl scipy
```

```python
import pandas as pd

# as consultas
q = pd.read_parquet("data/perguntas_192.parquet")

# o contraste central, consulta a consulta (2 304 linhas)
a = pd.read_csv("results/analise/auditoria_query_a_query.csv")

# os dois resultados de primeira ordem
a.classificacao_D.value_counts(normalize=True)   # 88,50 % de empates
a.eta.median()                                   # 0,9552
```

> **Leia as contagens por direção, não as médias.** A média de η é negativa em
> 6 das 12 células enquanto a mediana supera 0,94 em todas — as duas descrevem
> populações diferentes, e a explicação está na Seção 6.3 do artigo. Pares
> censurados (`classificacao_D == "nao_comparavel"`) nunca são empates.

---

## Sobre o que não está aqui

O **texto integral dos documentos normativos** não é disponibilizado. A sua
divulgação depende de autorização institucional e não foi objeto do
consentimento obtido na aprovação ética do estudo — e é exatamente a restrição
que motiva o trabalho, já que é por não poder sair da instituição que o corpus
exige modelos executados localmente.

Isso tem duas consequências, e a segunda costuma passar despercebida:

1. `corpus_index_*.parquet` traz os identificadores, os vetores densos e os
   comprimentos de cada segmento, mas **não** as colunas de texto;
2. **as respostas geradas pelos modelos também não são distribuídas.** Elas
   eram ancoradas nos documentos e reproduziam trechos literais do corpus —
   medido em janelas de 12 palavras, as 6 912 respostas recuperariam cerca de
   **20 % do texto dos 740 segmentos**, e algumas dezenas de segmentos quase
   por inteiro. Publicá-las contornaria na prática a mesma restrição que a
   alínea 1 respeita. As colunas `resposta_final` (Bloco 1) e
   `texto_resposta_final` (Bloco 2) foram removidas dos arquivos de resultado.

Nada do que o artigo reporta depende do texto removido: BERTScore em precisão,
revocação e F1, conformidade de formato, profundidade de parada e contagem de
*tokens* foram calculados antes e permanecem nos arquivos. Pesquisadores que
precisem do texto-fonte ou das saídas dos sistemas para fins de replicação podem
contatar o autor correspondente; o acesso depende de autorização da instituição
de origem.

Também ficam de fora, deliberadamente: credenciais de acesso a repositórios de
pesos e os pesos dos modelos (todos públicos nos seus repositórios de origem).

O código de execução em `code/execucao/` é uma **cópia congelada** da versão
final usada nas execuções reportadas. O histórico de desenvolvimento ficou no
repositório de trabalho privado da equipe; os manifestos em `results/env/`
amarram cada arquivo de resultado ao ponto do código que o produziu.

---

## Proveniência

| Braço | *Commit* do código | Empacotado em |
|---|---|---|
| Recuperação limitada (`e5small`) | `c196c1b` | 10/09/2026 |
| Recuperação competente (`qwen3`) | `d26caf0` | 13/09/2026 |

Ambiente comum às seis execuções: GPU NVIDIA GeForce RTX 5090 (32 GB);
Python 3.12.3, `torch` 2.14.0+cu132, `bitsandbytes` 0.50.2, `numpy` 2.5.2,
`pandas` 3.0.5; semente base 42; *D*<sub>max</sub> = 50; teto de geração de 800
*tokens*. As divergências entre execuções — teto de *prompt*, versão da
biblioteca de inferência, rotina de atenção — estão registradas em
`results/env/` e discutidas no Online Resource 1 do artigo.

Os manifestos de cada entrega trazem os digestos SHA-256 de todos os arquivos
de resultado. `CHECKSUMS.sha256` permite verificar que os arquivos deste
repositório são os mesmos.

---

## Licenças

- **Código** — [MIT](LICENSE)
- **Dados e resultados** — [CC BY 4.0](LICENSE-DATA)

Ao reutilizar, cite o artigo e este repositório (ver [CITATION.cff](CITATION.cff)).

---

## Ética

A etapa de validação por especialistas foi aprovada pelo comitê de ética em
pesquisa da instituição (CAAE 88037225.9.0000.5317), e todos os 12
participantes forneceram consentimento informado. O corpus é de documentação
normativa administrativa: não contém dados de pacientes.
