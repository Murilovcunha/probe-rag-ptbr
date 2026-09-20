"""
00_preparar_dados.py -- Converte os CSV originais em Parquet e valida o
mapeamento do chunk-ouro.

Por que Parquet e não CSV:
  * o corpus tem 740 embeddings de 4096 dimensões guardados como TEXTO JSON
    no CSV (42 MB). Cada leitura exige json.loads() em 740 strings gigantes:
    ~8 s por execução, repetidos em toda rodada de todo modelo. Em Parquet os
    vetores viram listas de float32 nativas e a leitura cai para ~0,2 s;
  * o arquivo cai de 42 MB para ~13 MB, o que também facilita o envio;
  * tipos são preservados (nada de "id" virar float por causa de um NaN).

O que este script produz em dados/:
  corpus_export.parquet     id, nome_arquivo, texto_artigo, texto_pagina,
                            numero_tokens, vector (list<float32>, dim 4096)
  perguntas_192.parquet     dataset híbrido do Bloco 1 (192 perguntas)
  perguntas_74_gold.parquet dataset sintético do Bloco 2 (74 perguntas) com
                            id_chunk_gold e ids_chunk_gold_todos

MAPEAMENTO DO CHUNK-OURO (verificado, ver README seção "Decisões"):
  no gabarito_74perguntas, a coluna que aponta para o corpus é `id_resposta`,
  no formato "173 e 174 e 176 e 180" (um ou mais ids separados por " e ").
  As colunas `id` e `chunk_id` do gabarito NÃO batem com corpus.id -- confirmam
  o nome do arquivo em apenas 3 das 74 linhas (coincidência), enquanto
  `id_resposta` confere o nome do arquivo em 74/74. O script re-executa essa
  validação e aborta se ela deixar de valer.

Uso:
  python 00_preparar_dados.py --entrada csv_originais --saida dados
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

NOME_CORPUS_CSV = "texto_qwen3_embedding.csv"
NOME_192_CSV = "perguntas_respostas_unificadas.csv"
NOME_74_CSV = "gabarito_74perguntas_202609012215.csv"


def parse_ids(valor) -> list[int]:
    """'173 e 174 e 176 e 180' -> [173, 174, 176, 180]. Robusto a ',', '/', ';'."""
    return [int(x) for x in re.findall(r"\d+", str(valor))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="csv_originais",
                    help="pasta com os 3 CSV originais")
    ap.add_argument("--saida", default="dados",
                    help="pasta de destino dos .parquet")
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    p = lambda nome: os.path.join(args.entrada, nome)  # noqa: E731

    for nome in (NOME_CORPUS_CSV, NOME_192_CSV, NOME_74_CSV):
        if not os.path.exists(p(nome)):
            sys.exit(f"ERRO: não encontrei {p(nome)}")

    # ------------------------------------------------------------------
    # 1. Corpus
    # ------------------------------------------------------------------
    print(f"Lendo {NOME_CORPUS_CSV} (isso leva ~10 s por causa dos embeddings em texto)...")
    corpus = pd.read_csv(p(NOME_CORPUS_CSV))
    esperadas = {"id", "nome_arquivo", "texto_artigo", "vector", "numero_tokens", "texto_pagina"}
    faltando = esperadas - set(corpus.columns)
    if faltando:
        sys.exit(f"ERRO: colunas ausentes no corpus: {faltando}")

    print("Convertendo embeddings de JSON texto para float32...")
    vetores = [np.asarray(json.loads(v), dtype=np.float32) for v in corpus["vector"]]
    dims = {len(v) for v in vetores}
    if len(dims) != 1:
        sys.exit(f"ERRO: embeddings com dimensões diferentes: {sorted(dims)}")
    dim = dims.pop()
    normas = np.array([float(np.linalg.norm(v)) for v in vetores])

    corpus["vector"] = [v.tolist() for v in vetores]
    corpus["id"] = corpus["id"].astype("int64")
    corpus["len_texto_artigo"] = corpus["texto_artigo"].astype(str).str.len()
    corpus["len_texto_pagina"] = corpus["texto_pagina"].astype(str).str.len()

    if not corpus["id"].is_unique:
        sys.exit("ERRO: corpus.id não é único -- o ranking e o gabarito dependem disso.")

    print(f"  {len(corpus)} chunks | dim={dim} | norma média={normas.mean():.4f} "
          f"(esperado ~1.0, embeddings já normalizados)")
    print(f"  {corpus['nome_arquivo'].nunique()} arquivos-fonte | "
          f"{corpus['texto_pagina'].nunique()} páginas distintas em texto_pagina")
    print(f"  numero_tokens (texto_artigo): média {corpus['numero_tokens'].mean():.0f}, "
          f"máx {corpus['numero_tokens'].max()}")

    destino_corpus = os.path.join(args.saida, "corpus_export.parquet")
    corpus.to_parquet(destino_corpus, index=False)
    print(f"  -> {destino_corpus} "
          f"({os.path.getsize(destino_corpus) / 1e6:.1f} MB, era "
          f"{os.path.getsize(p(NOME_CORPUS_CSV)) / 1e6:.1f} MB em CSV)")

    # ------------------------------------------------------------------
    # 2. Dataset híbrido de 192 perguntas (Bloco 1)
    # ------------------------------------------------------------------
    print(f"\nLendo {NOME_192_CSV}...")
    q192 = pd.read_csv(p(NOME_192_CSV))
    q192 = q192.rename(columns={"resposta": "resposta_gabarito"})
    q192["pergunta"] = q192["pergunta"].astype(str)
    q192["resposta_gabarito"] = q192["resposta_gabarito"].fillna("").astype(str)
    q192 = q192.reset_index(drop=True)
    q192["pergunta_idx"] = np.arange(len(q192))

    destino_192 = os.path.join(args.saida, "perguntas_192.parquet")
    q192.to_parquet(destino_192, index=False)
    print(f"  {len(q192)} perguntas -> {destino_192}")

    # ------------------------------------------------------------------
    # 3. Dataset sintético de 74 perguntas com chunk-ouro (Bloco 2)
    # ------------------------------------------------------------------
    print(f"\nLendo {NOME_74_CSV}...")
    q74 = pd.read_csv(p(NOME_74_CSV))
    q74["pergunta"] = q74["pergunta"].astype(str)
    q74 = q74.rename(columns={"resposta": "resposta_gabarito"})
    q74["resposta_gabarito"] = q74["resposta_gabarito"].fillna("").astype(str)

    q74["ids_chunk_gold_todos"] = q74["id_resposta"].apply(parse_ids)

    # ---- validação do mapeamento -------------------------------------
    mapa_id_arquivo = dict(zip(corpus["id"], corpus["nome_arquivo"]))
    ids_corpus = set(corpus["id"])

    fora = [ids for ids in q74["ids_chunk_gold_todos"]
            if any(i not in ids_corpus for i in ids)]
    if fora:
        sys.exit(f"ERRO: {len(fora)} linhas do gabarito citam ids inexistentes no corpus: {fora[:3]}")

    confere = [
        row["nome_arquivo"] in {mapa_id_arquivo[i] for i in row["ids_chunk_gold_todos"]}
        for _, row in q74.iterrows()
    ]
    n_ok = int(np.sum(confere))
    print(f"  validação id_resposta -> nome_arquivo: {n_ok}/{len(q74)} conferem")
    if n_ok < len(q74):
        divergentes = q74.loc[[not c for c in confere], ["pergunta", "id_resposta", "nome_arquivo"]]
        print(divergentes.to_string())
        sys.exit("ERRO: o mapeamento do chunk-ouro não fecha. NÃO rode os experimentos assim -- "
                 "avise o Murilo antes.")

    # id_chunk_gold principal = primeiro id declarado no gabarito
    q74["id_chunk_gold"] = q74["ids_chunk_gold_todos"].apply(lambda x: int(x[0]))
    q74["n_golds"] = q74["ids_chunk_gold_todos"].apply(len)
    q74 = q74.reset_index(drop=True)
    q74["pergunta_idx"] = np.arange(len(q74))

    colunas = ["pergunta_idx", "pergunta", "resposta_gabarito", "nome_arquivo",
               "id_chunk_gold", "ids_chunk_gold_todos", "n_golds",
               "id_resposta", "cosine_similarity"]
    colunas = [c for c in colunas if c in q74.columns]
    destino_74 = os.path.join(args.saida, "perguntas_74_gold.parquet")
    q74[colunas].to_parquet(destino_74, index=False)
    print(f"  {len(q74)} perguntas -> {destino_74}")
    print(f"  perguntas com mais de um chunk-ouro: {int((q74['n_golds'] > 1).sum())} "
          f"(baseline de acaso do Bloco 2 = n_golds/10, não 1/10)")

    print("\nOK. Copie a pasta 'dados/' inteira junto com os scripts.")


if __name__ == "__main__":
    main()
