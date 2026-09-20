# Dicionário de dados

Todos os arquivos desta pasta estão sob [CC BY 4.0](../LICENSE-DATA).
Os digestos SHA-256 estão em `CHECKSUMS.sha256`.

---

## `perguntas_192.parquet` — o conjunto de avaliação (192 linhas)

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `pergunta_idx` | int | Índice de 0 a 191. **Chave de junção com todos os arquivos de resultado.** |
| `id` | int | Identificador de origem, herdado da planilha de construção |
| `pergunta` | str | A consulta, tal como submetida ao sistema |
| `resposta_gabarito` | str | Resposta de referência, usada pelo BERTScore |
| `arquivo_fonte` | str | Documento normativo de origem |

**Os pares estão ordenados por origem.** `pergunta_idx` 0–73 é o subconjunto
sintético — questões geradas a partir do segmento que as responde, portanto
verbosas, bem formadas e ricas em termos do segmento. `pergunta_idx` 74–191 é
o subconjunto de especialistas — questões formuladas por 12 profissionais da
instituição, curtas, às vezes vagas ou elípticas, com erros de digitação.

Os dois subconjuntos têm comportamento mensuravelmente distinto (Seção 6.5 do
artigo: 94,71 % contra 84,60 % de empates em profundidade de parada; 1 contra
71 censuras incrementais). Qualquer amostra tomada pelas *n* primeiras
questões mede só o subconjunto fácil.

As respostas de referência do subconjunto sintético foram geradas com auxílio
de um modelo comercial em etapa anterior e independente da avaliação; as do
subconjunto de especialistas foram formuladas e validadas pelos profissionais.

---

## `perguntas_74_gold.parquet` — subconjunto com segmento-ouro (74 linhas)

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `pergunta_idx` | int | Junta com `perguntas_192.parquet` |
| `pergunta`, `resposta_gabarito`, `nome_arquivo` | str | Como acima |
| `id_chunk_gold` | int | Identificador do segmento portador da resposta |
| `ids_chunk_gold_todos` | array&lt;int&gt; | Todos os segmentos-ouro, quando há mais de um |
| `n_golds` | int | Quantidade de segmentos-ouro (1 a 4; 12 questões têm mais de um) |
| `id_resposta` | str | Forma textual da lista acima |
| `cosine_similarity` | float | Similaridade entre a consulta e o segmento-ouro na indexação |

É sobre estas 74 consultas que roda a sonda de incerteza (Bloco 2). Repare que
são exatamente o subconjunto sintético — nenhum achado da sonda se estende às
consultas dos profissionais, para as quais não há anotação de segmento
portador.

---

## `corpus_index_e5small.parquet` e `corpus_index_qwen3.parquet` — os 740 segmentos indexados, **sem texto**

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `id` | int | Identificador do segmento, referenciado por `id_chunk_gold` |
| `nome_arquivo` | str | Documento normativo de origem |
| `vector` | array&lt;float&gt; | Vetor denso do segmento (384 dimensões no `e5small`) |
| `numero_tokens` | int | Tamanho do segmento em *tokens* |
| `len_texto_artigo` | int | Comprimento, em caracteres, do campo de artigo |
| `len_texto_pagina` | int | Comprimento, em caracteres, do campo de página |

**As colunas `texto_artigo` e `texto_pagina` foram removidas.** Elas contêm o
texto integral dos documentos normativos, que não é liberado. Os comprimentos
são mantidos porque a análise de sensibilidade do BERTScore ao comprimento da
resposta depende deles.

A coluna de texto efetivamente usada nos experimentos é a de **página**, não a
de artigo — registrado no metadado de cada execução. Os 740 segmentos apontam
para apenas 462 páginas distintas, de modo que percorrer *k* posições do
*ranking* não injeta *k* unidades de texto distintas.

---

## `embeddings_perguntas_*.parquet` — vetores densos das consultas

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `pergunta_idx` | int | Junta com `perguntas_192.parquet` |
| `embedding_pergunta` | array&lt;float&gt; | Vetor denso da consulta |

Calculados uma única vez e congelados em disco, o que garante *ranking*
idêntico bit a bit entre as arquiteturas comparadas. O sufixo do arquivo
identifica o recuperador.

Sobre a convenção de codificação do `multilingual-e5-small`: o índice foi
construído **sem** prefixo de consulta/passagem. Empregar o prefixo em apenas
um dos lados reduz o MRR em 0,157 — o Online Resource 1 do artigo detalha como
isso foi verificado por medição.

---

## Como os arquivos se juntam

```
perguntas_192.parquet ──pergunta_idx──┬── auditoria_query_a_query.csv   (2 304 linhas)
                                      ├── consolidado_bloco1_*.csv
                                      └── embeddings_perguntas_*.parquet

perguntas_74_gold.parquet ──id_chunk_gold── corpus_index_*.parquet
          └──pergunta_idx── consolidado_bloco2_*.csv, postos_bloco2_*.csv
```

---

## Uma nota sobre os vetores densos

Os vetores são liberados por serem necessários para reproduzir a recuperação e
as análises de profundidade. Registre-se, para quem for reutilizá-los, que
métodos de inversão de *embeddings* podem recuperar parcialmente o texto de
origem. O corpus é de documentação normativa administrativa — não contém dados
de pacientes —, mas a confidencialidade institucional continua valendo, e o uso
dos vetores para tentar reconstruir o texto dos documentos não está autorizado
pela licença nem pela aprovação ética do estudo.
