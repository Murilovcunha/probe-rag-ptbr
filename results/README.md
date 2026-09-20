# Arquivos de resultado

Todos os arquivos desta pasta estão sob [CC BY 4.0](../LICENSE-DATA).
Os digestos SHA-256 estão em `CHECKSUMS.sha256`, e os manifestos de cada
entrega, em `env/`, amarram cada arquivo ao ponto do código que o produziu.

A chave de junção com `data/perguntas_192.parquet` é `pergunta_idx` em todos
os arquivos.

> **Sobre as respostas geradas.** As colunas com o texto das respostas
> produzidas pelos modelos (`resposta_final` no Bloco 1, `texto_resposta_final`
> no Bloco 2) **não são distribuídas**. Elas eram ancoradas nos documentos
> normativos e reproduziam trechos literais do corpus, cuja divulgação depende
> de autorização institucional (ver [Sobre o que não está
> aqui](../README.md#sobre-o-que-não-está-aqui)). Tudo o que foi calculado a
> partir delas — BERTScore em precisão, revocação e F1, conformidade de
> formato, profundidade de parada, contagem de *tokens* — permanece nos
> arquivos, de modo que nenhuma análise reportada no artigo depende do texto
> removido. Pesquisadores que precisem das saídas dos sistemas para replicação
> podem contatar o autor correspondente.

---

## `bloco1/` — o laço de parada declarada

### `consolidado_bloco1_e5small.csv` e `consolidado_bloco1_qwen3.csv`

Uma linha por (consulta × gerador × modalidade × modo de entrega): 3 456 linhas
por recuperador. O sufixo identifica o recuperador.

**Identificação da célula**

| Coluna | Conteúdo |
|---|---|
| `modelo` | gerador |
| `retriever` | `e5small` ou `qwen3` |
| `criterio` | modalidade de parada: `A` (autômato) ou `B` (contrato JSON) |
| `modo` | `iterativo`, `incremental` ou `fixo50` |
| `pergunta_idx` | 0 a 191 |
| `pergunta`, `resposta_gabarito`, `arquivo_fonte` | repetidos de `data/perguntas_192.parquet`, por conveniência |

**Profundidade de parada ($D$)**

| Coluna | Conteúdo |
|---|---|
| `D_iter` | posição do *ranking* em que o gerador declarou suficiência. **Nulo quando censurado** |
| `D_iter_regra_alternativa` | onde o laço teria parado sob a regra "sentinela **ou** recusa em prosa" (ver [decisões técnicas](../code/execucao/DECISOES_TECNICAS.md#22-a-sentinela-exata-e-a-regra-alternativa-calculada-de-graça)) |
| `D_chunk` | segmentos distintos efetivamente injetados no contexto |
| `id_chunk_parada` | identificador do segmento da posição de parada |
| `parou_no_gold` | a posição de parada é um segmento-ouro (só nas 74 rastreáveis) |
| `censurado_direita` | esgotou $D_{max}=50$ sem resposta válida |
| `censurado_limite_tokens` | encerrado por estouro do teto de 32 768 *tokens* de entrada |
| `contexto_truncado` | contexto encurtado para caber (ocorre no `fixo50`) |
| `n_paginas_puladas` | posições puladas por estouro, no modo iterativo |

**Censura nunca é empate e nunca é $D=50$.** As médias de $D$ se calculam
apenas sobre as não censuradas, e a contagem de censuras é reportada à parte.

**Custo**

| Coluna | Conteúdo |
|---|---|
| `n_inferencias` | chamadas ao gerador até a parada |
| `tokens_prompt_acum`, `tokens_total_acum` | *tokens* de entrada e totais |
| `n_paginas_enviadas` | páginas distintas que chegaram ao contexto |
| `tempo_s`, `vram_pico_gb` | tempo de parede e pico de memória por consulta |

A eficiência $\eta$ **não** se calcula daqui por diferença entre modos: ela é
$1 - T_{\text{incremental}}/T_{\text{fixo50}}$, está pronta em
`analise/auditoria_query_a_query.csv`, e o comparador é sempre o `fixo50`.

**Fidelidade e formato**

| Coluna | Conteúdo |
|---|---|
| `S_bertscore_p`, `S_bertscore_r`, `S_bertscore_f1` | BERTScore da resposta contra a referência, decomposto |
| `n_iter_conformes` | iterações em que a chave JSON foi lida |
| `n_iter_json_estrito` | iterações cuja saída **já era** o objeto pedido, sem nada em volta |
| `n_iter_formato_invalido` | iterações que caíram no autômato por formato quebrado |
| `n_iter_max_tokens` | iterações interrompidas pelo teto de geração |
| `similaridade_na_parada`, `similaridade_top1`, `id_chunk_top1` | do *ranking* |
| `id_chunk_gold`, `ids_chunk_gold_todos` | anotação, só nas 74 rastreáveis |

O BERTScore está decomposto porque precisão e revocação respondem de forma
oposta ao comprimento da resposta, e o F1 sozinho esconde isso.

### `tabela_bloco1_*.csv`

As 12 células agregadas por (gerador × modalidade × modo), como entram nas
tabelas do artigo.

### `direcao_iter_vs_incremental.csv`

As contagens por direção do contraste central. **É a leitura primária**: as
médias de $\delta D$ e de $\eta$ cancelam sinais opostos e descrevem população
diferente da que as contagens descrevem.

---

## `bloco2/` — a sonda de incerteza

444 linhas por recuperador: 74 consultas × 3 geradores × 2 modalidades. Só o
subconjunto sintético entra, porque só ele tem segmento-ouro anotado.

### `consolidado_bloco2_*.csv`

| Grupo | Colunas |
|---|---|
| Célula | `modelo`, `retriever`, `criterio`, `pergunta_idx`, `pergunta`, `resposta_gabarito` |
| Anotação | `id_chunk_gold`, `ids_chunk_gold_todos`, `n_golds_no_top10` |
| Posição do ouro | `posicao_gold_no_ranking_natural`, `posicoes_todos_golds_natural`, `posicao_gold_no_top10_final`, `posicoes_golds_no_top10` |
| Estratificação | `gold_foi_reposicionado_no_top10`, `id_chunk_removido_na_substituicao`, `estrato` |
| Parada simulada | `D_embaralhado`, `D_embaralhado_regra_alternativa`, `id_chunk_D`, `acertou_chunk_gold`, `n_posicoes_validas` |
| Entropia | `posicao_min_entropia`, `entropia_minima`, `min_entropia_e_gold`, `concordancia_D_entropia` |
| Comparador | `max_similaridade_e_gold` |
| Formato | `n_pos_conformes`, `n_pos_json_estrito`, `n_pos_formato_invalido`, `n_pos_janela_json_alinhada` |
| Custo e fidelidade | `n_inferencias`, `eta_tokens_ate_D`, `eta_tokens_total_10`, `tempo_segundos`, `vram_pico_gb`, `S_bertscore_*` |

Três advertências que mudam a leitura destes números:

1. **`max_similaridade_e_gold` é o Precision@1 do recuperador**, não um
   comparador com variância própria — é invariante ao gerador por construção,
   e serve de verificação de sanidade. Por isso a estratificação em
   `estrato`;
2. **`min_entropia_e_gold` é secundária.** A medida principal é o **posto** do
   segmento-ouro entre as 10 posições, em `postos_bloco2_*.csv`, porque é uma
   medida por consulta ($n=74$ independentes) e é invariante a reescala
   monotônica;
3. **posições com `n_pos_janela_json_alinhada` baixa saem das médias de
   entropia.** Sob o contrato JSON os primeiros tokens são sintaxe, e a janela
   é deslocada para o início do valor; quando o formato quebra, não há para
   onde deslocar.

### `postos_bloco2_*.csv`

O posto do segmento-ouro sob cada escore — entropia, similaridade, *teacher
forcing*, sonda sim/não e a sobreposição lexical de controle. **É daqui que sai
o enunciado principal da sonda.** Regra multi-ouro: melhor posto entre os ouros.

### `nula_exata_*.csv` e `nula_estratificada_*.csv`

A nula em forma fechada e a nula por permutação dentro do estrato de respostas
válidas. As duas existem porque dão respostas diferentes: a segunda leva em
conta que uma parte da concordância aparente vem de o argmin cair sobre uma
recusa.

### `circularidade_*.csv`

O diagnóstico que mede quanto do sinal de entropia é, na verdade, um detector
de recusa: a fração de casos em que o argmin de entropia é uma recusa, e a AUC
de recusa contra resposta válida.

### `comparacao_modalidades_*.csv` e `modalidade_b_formato_*.csv`

Contraste A × B e as taxas de conformidade de formato. O *nível* de entropia
das duas modalidades não é comparável nem depois do alinhamento da janela; o
que se compara é o posto e o $D$.

---

## `analise/`

### `auditoria_query_a_query.csv` — 2 304 linhas, 42 colunas

O contraste incremental × iterativo consulta a consulta, em precisão plena.
**É o arquivo autoritativo**: as planilhas trazem os mesmos números
arredondados, e recomputar a partir delas dá resultados diferentes.

Uma linha por (recuperador × gerador × modalidade × consulta). Para cada eixo —
$D$, F1, precisão, revocação — há `*_incremental`, `*_iterativo`, `delta_*` e
`classificacao_*`. As colunas de custo trazem os três modos
(`tokens_incremental`, `tokens_iterativo`, `tokens_fixo50`), mais `eta`,
`classificacao_eta` e `eta_iterativo`.

As `classificacao_*` são a leitura autoritativa das direções:

```python
a.classificacao_D.value_counts()
# empate                       2039
# nao_comparavel                135   <- censurados: NUNCA são empates
# iterativo_gt_incremental       88
# incremental_gt_iterativo       42
```

As seis colunas `censurado_*` dizem, por modo, se a censura foi por $D_{max}$
ou por teto de *tokens* — é o que permite verificar se uma profundidade foi
truncada por memória em vez de por ancoragem.

### `resultados_completos_experimento.xlsx` — 24 abas

A consolidação em planilha, com o dicionário de colunas (`Dicionario`), as
verificações aritméticas (`Verificacao_Matematica`) e o registro de validação
(`Log_Validacao`). Os valores estão **arredondados para exibição** — para
recomputar qualquer coisa, use `auditoria_query_a_query.csv`.

---

## `env/`

Metadados de ambiente de cada uma das seis execuções (`metadata_b*.json`),
`requirements.lock`, os manifestos com os digestos SHA-256 de cada entrega e a
nota de correção da rodada de 13/09/2026. As divergências entre execuções —
teto de *prompt*, versão da biblioteca de inferência, rotina de atenção —
estão registradas aqui e discutidas no Online Resource 1 do artigo.
