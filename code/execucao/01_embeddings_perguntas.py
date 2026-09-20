"""
01_embeddings_perguntas.py -- Calcula UMA VEZ o embedding das perguntas com
Qwen3-Embedding-8B e salva em Parquet.

Por que isto é um script separado, e não uma "Fase 0" dentro de cada
experimento (como no experimento.py original):

  1. VRAM. O Qwen3-Embedding-8B ocupa ~16 GB em fp16. Carregá-lo no mesmo
     processo em que depois entra um LLM de 27B em 4-bit (~17 GB) deixa a
     5090 (32 GB) na borda do OOM. Mesmo com del + empty_cache, a
     fragmentação do allocator costuma sobrar e derrubar a rodada na
     primeira pergunta com contexto longo.
  2. Tempo. São 6 modelos no Bloco 1 e 4 no Bloco 2 = 10 rodadas. Recalcular
     os mesmos embeddings 10 vezes é desperdício puro.
  3. Determinismo. Os embeddings ficam congelados em disco, então o ranking
     é bit-a-bit idêntico entre todos os modelos avaliados -- que é
     exatamente a condição experimental exigida pela tese (o componente de
     recuperação permanece constante entre as condições).

Decisão de simetria (herdada do experimento.py): no qwen3 as perguntas são
codificadas SEM prefixo de instrução, do mesmo modo que os chunks do corpus
foram codificados. Trocar isso quebraria a comparabilidade com o corpus já
indexado.

Desde 06/09/2026 o prefixo é PROPRIEDADE DO RETRIEVER (comum.RETRIEVERS), e
não uma constante: os modelos multilingual-e5 são treinados com "query: " e
"passage: ". Rode o 06_diagnostico_retriever.py antes -- ele compara as duas
convenções contra o MRR já publicado e diz qual reproduz o corpus.

Uso:
  python 01_embeddings_perguntas.py
  LOGPROB_RETRIEVER=e5small python 01_embeddings_perguntas.py
"""

import os

import numpy as np
import pandas as pd
import torch

import comum


def carregar_modelo_embedding():
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {comum.EMBED_MODEL_ID} em {device}...")
    return SentenceTransformer(comum.EMBED_MODEL_ID, device=device)


def codificar(modelo, perguntas):
    # Prefixo da convenção do retriever (vazio no qwen3). Os modelos
    # multilingual-e5 são TREINADOS com "query: " / "passage: "; usar de um
    # lado só degrada a recuperação por um motivo que não é o fator em
    # estudo. Qual convenção usar sai do 06_diagnostico_retriever.py, que
    # compara as duas contra o número já publicado.
    pref = comum.PREFIXO_PERGUNTA
    if pref:
        print(f"  prefixo aplicado às perguntas: {pref!r}")
    return modelo.encode(
        [pref + str(p) for p in perguntas],
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embeddings_falsos(perguntas, dim):
    """
    Embeddings pseudoaleatórios determinísticos, só para a flag --smoke.
    Permite validar todo o pipeline (ranking, shuffle, regex, parquets) sem
    baixar os 16 GB do Qwen3-Embedding-8B. NÃO usar em rodada de verdade.
    """
    saida = []
    for i, p in enumerate(perguntas):
        rng = np.random.RandomState(1000 + i)
        v = rng.normal(size=dim).astype(np.float32)
        saida.append(v / np.linalg.norm(v))
        del p
    return np.stack(saida)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="gera embeddings falsos determinísticos, sem baixar modelo")
    args = ap.parse_args()

    os.makedirs(comum.DIR_DADOS, exist_ok=True)

    tarefas = [
        (comum.ARQ_PERGUNTAS_192, comum.ARQ_EMB_192, "192 perguntas (Bloco 1)"),
        (comum.ARQ_PERGUNTAS_74, comum.ARQ_EMB_74, "74 perguntas (Bloco 2)"),
    ]

    pendentes = [(e, s, d) for e, s, d in tarefas if not os.path.exists(s)]
    for entrada, _, desc in tarefas:
        if not os.path.exists(entrada):
            raise SystemExit(f"ERRO: {entrada} não existe. Rode 00_preparar_dados.py primeiro.")
        del desc

    if not pendentes:
        print("Embeddings já existem. Nada a fazer.")
        print("(apague os arquivos em dados/embeddings_*.parquet para recalcular)")
        return

    if args.smoke:
        print("[SMOKE] gerando embeddings FALSOS -- não use os resultados.")
        dim = len(pd.read_parquet(comum.ARQ_CORPUS)["vector"].iloc[0])
        modelo = None
    else:
        modelo = carregar_modelo_embedding()
    try:
        for entrada, saida, descricao in pendentes:
            df = pd.read_parquet(entrada)
            print(f"\n{descricao}: codificando {len(df)} perguntas...")
            perguntas = df["pergunta"].astype(str).tolist()
            embs = (embeddings_falsos(perguntas, dim) if args.smoke
                    else codificar(modelo, perguntas))
            out = pd.DataFrame({
                "pergunta_idx": df["pergunta_idx"].to_numpy(),
                "pergunta": df["pergunta"].astype(str).to_numpy(),
                "embedding_pergunta": [e.astype(np.float32).tolist() for e in embs],
            })
            out.to_parquet(saida, index=False)
            print(f"  -> {saida}  (dim={embs.shape[1]})")
    finally:
        # libera os ~16 GB do modelo de embedding antes de qualquer outra coisa
        del modelo
        comum.limpar_vram()
        print("\nModelo de embedding descarregado da VRAM.")

    print("Pronto. Agora rode 02_bloco1_desempenho.py e 03_bloco2_entropia.py.")


if __name__ == "__main__":
    main()
