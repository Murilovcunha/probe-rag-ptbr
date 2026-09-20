"""
06_diagnostico_retriever.py -- O retriever tem variância a oferecer?
Roda em CPU, em ~1 min, SEM GPU e SEM LLM nenhum.

Por que existe. A seção 9.24 mostrou que, com o qwen3, `D = 1` em 19 de 20
perguntas: o contraste iterativo x incremental do Bloco 1 não tem onde
acontecer. A seção 9.25 propôs trocar o embedder por um mais fraco
(`multilingual-e5-small`) para devolver variância a `D`. A pergunta "isso vai
funcionar?" custaria 55-110 h de GPU para responder por tentativa.

Não precisa. O que decide é a POSIÇÃO DO CHUNK-OURO NO RANKING, e ela sai de
um produto de matrizes: se o gold está em 3º, o método precisa rejeitar dois
chunks antes de chegar nele, e `D >= 3` -- a menos que algum chunk anterior
também sustente resposta. Ou seja: a distribuição do posto do gold é um
LIMITE SUPERIOR direto para a variância de `D`, calculável sem gerar um token.

O script também resolve, POR MEDIDA, a pegadinha dos prefixos do e5. Os
modelos `multilingual-e5` são treinados com `"query: "` e `"passage: "`. Se as
perguntas forem codificadas com uma convenção e o corpus com outra, a
recuperação piora -- e pioraria pelo motivo errado, contaminando a comparação
com o qwen3. Como não dá para inspecionar o corpus e descobrir a convenção
dele, o script codifica as perguntas DAS DUAS FORMAS e compara cada uma com o
número já publicado (Tabela 2 do PROPOR 2026). A convenção certa é a que
reproduz o artigo.

Uso:
  LOGPROB_RETRIEVER=e5small python 06_diagnostico_retriever.py
  LOGPROB_RETRIEVER=e5small python 06_diagnostico_retriever.py --gravar com
  python 06_diagnostico_retriever.py                 # baseline do qwen3
  python 06_diagnostico_retriever.py --smoke         # só exercita a lógica
"""

import argparse
import os

import numpy as np
import pandas as pd

import comum


def _mrr_e_postos(matriz, embs_perguntas, ids_corpus, golds_por_pergunta):
    """
    Para cada pergunta: posto (1-based) do MELHOR chunk-ouro no ranking
    completo, e o recíproco desse posto (MRR por consulta).
    """
    postos = []
    for emb, ids_gold in zip(embs_perguntas, golds_por_pergunta):
        ordem, _ = comum.ranquear(emb, matriz, k=None)
        pos = np.where(np.isin(ids_corpus[ordem], list(ids_gold)))[0]
        postos.append(int(pos[0]) + 1 if len(pos) else len(ids_corpus))
    p = np.asarray(postos, dtype=float)
    return p, 1.0 / p


def _relatorio(nome, postos, rr, ref_mrr=None, ref_freq=None):
    freq1 = 100 * float((postos == 1).mean())
    print(f"\n  [{nome}]")
    print(f"    MRR médio            {rr.mean():.4f}"
          + (f"   (artigo: {ref_mrr:.4f})" if ref_mrr else ""))
    print(f"    MRR = 1 (gold em 1º) {freq1:.2f}%"
          + (f"   (artigo: {ref_freq:.2f}%)" if ref_freq else ""))
    print(f"    posto do gold: mediana {np.median(postos):.0f}  "
          f"p75 {np.quantile(postos, .75):.0f}  p90 {np.quantile(postos, .90):.0f}  "
          f"máx {postos.max():.0f}")
    faixas = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10 ** 9)]
    rot = ["1º", "2º-3º", "4º-10º", "11º-50º", "> 50º"]
    print("    distribuição:", "  ".join(
        f"{r}={100 * float(((postos >= a) & (postos <= b)).mean()):.1f}%"
        for (a, b), r in zip(faixas, rot)))
    return freq1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", choices=["com", "sem"], default=None,
                    help="grava os embeddings das perguntas com a convenção "
                         "escolhida ('com' = prefixo query:). Sem isto o script "
                         "só diagnostica e não escreve nada")
    ap.add_argument("--usar-parquet", nargs="*", metavar="TAG=ARQUIVO",
                    help="usa embeddings JÁ CALCULADOS em vez de carregar o "
                         "modelo, ex.: --usar-parquet sem=e_SEM.parquet "
                         "com=e_COM.parquet. Serve quando quem gerou os "
                         "embeddings foi outra máquina -- que é o caso normal, "
                         "porque o diagnóstico não precisa de GPU e a geração "
                         "precisa do modelo baixado")
    ap.add_argument("--smoke", action="store_true",
                    help="usa embeddings falsos: exercita a lógica sem baixar modelo")
    args = ap.parse_args()

    cfg = comum.CFG_RETRIEVER
    print("=" * 74)
    print(f"DIAGNÓSTICO DO RETRIEVER -- {comum.RETRIEVER} ({comum.EMBED_MODEL_ID})")
    print("=" * 74)
    print("Sem GPU, sem LLM. O que se mede é a posição do chunk-ouro no ranking,")
    print("que é o limite superior da variância que `D` pode ter no Bloco 1.")

    corpus = pd.read_parquet(comum.ARQ_CORPUS)
    matriz = comum.matriz_corpus(corpus)
    ids_corpus = corpus["id"].to_numpy()
    p74 = pd.read_parquet(comum.ARQ_PERGUNTAS_74)
    golds = [{int(x) for x in r} for r in p74["ids_chunk_gold_todos"]]
    print(f"\ncorpus: {len(corpus)} chunks, dim {matriz.shape[1]} | "
          f"perguntas com gabarito: {len(p74)}")

    # ---- codifica as perguntas nas DUAS convenções -------------------
    variantes = {}
    if args.usar_parquet:
        for item in args.usar_parquet:
            if "=" not in item:
                raise SystemExit(f"esperava TAG=ARQUIVO, recebi {item!r}")
            tag, arq = item.split("=", 1)
            d = pd.read_parquet(arq).set_index("pergunta_idx").loc[
                p74["pergunta_idx"].to_numpy()]
            variantes[tag.strip().lower()] = np.stack(
                d["embedding_pergunta"].to_numpy()).astype(np.float32)
            print(f"  [{tag}] {arq} -> {variantes[tag.strip().lower()].shape}")
    elif args.smoke:
        rng = np.random.default_rng(7)
        for tag in ("sem", "com"):
            e = rng.normal(size=(len(p74), matriz.shape[1])).astype(np.float32)
            variantes[tag] = e / np.linalg.norm(e, axis=1, keepdims=True)
        print("[SMOKE] embeddings falsos -- os números abaixo não significam nada.")
    else:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\ncarregando {comum.EMBED_MODEL_ID} em {dev}...")
        modelo = SentenceTransformer(comum.EMBED_MODEL_ID, device=dev)
        for pref, tag in ((("", "sem"), (cfg["prefixo_pergunta"], "com"))
                          if cfg["prefixo_pergunta"] else (("", "sem"),)):
            variantes[tag] = modelo.encode(
                [pref + str(t) for t in p74["pergunta"]], batch_size=32,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)

    # ---- compara as convenções contra o número publicado -------------
    print("\n" + "-" * 74)
    print("1. CONVENÇÃO DE PREFIXO -- qual reproduz o artigo?")
    print("-" * 74)
    resultados = {}
    for tag, embs in variantes.items():
        postos, rr = _mrr_e_postos(matriz, embs, ids_corpus, golds)
        freq1 = _relatorio(f"perguntas {'COM' if tag == 'com' else 'SEM'} prefixo "
                           f"{cfg['prefixo_pergunta']!r}", postos, rr,
                           cfg.get("mrr_medio_artigo"), cfg.get("freq_mrr1_artigo"))
        resultados[tag] = (postos, rr, freq1)

    alvo = cfg.get("freq_mrr1_artigo")
    escolhida = None
    if len(resultados) > 1 and alvo:
        escolhida = min(resultados, key=lambda t: abs(resultados[t][2] - alvo))
        outra = [t for t in resultados if t != escolhida][0]
        d1 = abs(resultados[escolhida][2] - alvo)
        d2 = abs(resultados[outra][2] - alvo)
        print(f"\n  -> a convenção '{escolhida}' fica a {d1:.2f} pontos do artigo; "
              f"a outra, a {d2:.2f}.")
        if abs(d1 - d2) < 2:
            print("  <<< as duas ficam perto: o teste NÃO distingue. Nesse caso")
            print("      pergunte a quem gerou o corpus qual prefixo foi usado, em")
            print("      vez de escolher pelo número.")
        else:
            print(f"      Isso identifica como o CORPUS foi indexado. Use '{escolhida}'"
                  f" -- e registre a escolha na tese.")
    elif len(resultados) == 1:
        escolhida = "sem"
        print("\n  (este retriever não declara prefixo; só há uma convenção)")

    # ---- o que isso projeta para o eixo D ----------------------------
    postos, rr, freq1 = resultados[escolhida or list(resultados)[0]]
    print("\n" + "-" * 74)
    print("2. O QUE ISSO PROJETA PARA O EIXO D DO BLOCO 1")
    print("-" * 74)
    fora_de_1 = 100 * float((postos > 1).mean())
    print(f"  gold FORA do 1º lugar: {fora_de_1:.1f}% das perguntas")
    print(f"  -> é o LIMITE SUPERIOR de P(D > 1). O valor realizado fica ABAIXO")
    print(f"     dele, porque um chunk não-ouro também pode sustentar resposta.")
    print(f"     Na Tabela 4 do artigo essa folga foi de 11 a 27 pontos.")
    print(f"  projeção de P(D > 1): entre ~{max(fora_de_1 - 27, 0):.0f}% e "
          f"~{fora_de_1:.0f}%")
    n192 = 192
    print(f"  em 192 perguntas: ~{n192 * max(fora_de_1 - 27, 0) / 100:.0f} a "
          f"~{n192 * fora_de_1 / 100:.0f} perguntas INFORMATIVAS por modelo")
    print("     ('informativa' = D pode passar de 1, que é a única situação em")
    print("      que iterativo e incremental têm como diferir)")
    print("\n  Referência medida em 06/09/2026 com o qwen3, 20 perguntas")
    print("  espalhadas: D = 1 em 19/20, ou seja ~5% informativas. Foi por isso")
    print("  que o contraste central não fechou.")
    if fora_de_1 < 15:
        print("\n  <<< ATENÇÃO: menos de 15% dos golds saem do 1º lugar. Trocar de")
        print("      retriever NÃO vai devolver variância a D -- não gaste GPU nisso.")
    elif fora_de_1 < 30:
        print("\n  [limítrofe] a variância aumenta, mas pouco. Vale rodar UM modelo")
        print("      espalhado antes de comprometer os seis.")
    else:
        print("\n  [ok] há variância suficiente para o contraste existir. O desenho")
        print("      parcial da seção 9.25 (6 modelos num retriever + 2-3 nos dois)")
        print("      passa a fazer sentido.")

    # ---- grava, se pedido --------------------------------------------
    if args.gravar:
        if args.gravar not in variantes:
            raise SystemExit(f"convenção {args.gravar!r} não foi calculada")
        if args.smoke:
            raise SystemExit("--gravar não faz sentido com --smoke")
        from sentence_transformers import SentenceTransformer
        import torch
        modelo = SentenceTransformer(comum.EMBED_MODEL_ID,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
        pref = cfg["prefixo_pergunta"] if args.gravar == "com" else ""
        for arq_perg, saida in ((comum.ARQ_PERGUNTAS_192, comum.ARQ_EMB_192),
                                (comum.ARQ_PERGUNTAS_74, comum.ARQ_EMB_74)):
            df = pd.read_parquet(arq_perg)
            e = modelo.encode([pref + str(t) for t in df["pergunta"]], batch_size=32,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
            pd.DataFrame({"pergunta_idx": df["pergunta_idx"].to_numpy(),
                          "embedding_pergunta": list(e.astype(np.float32))}
                         ).to_parquet(saida, index=False)
            print(f"  -> {saida}  ({e.shape}, prefixo={pref!r})")
        print("\nPronto. Agora o Bloco 1 e o Bloco 2 podem rodar com "
              f"LOGPROB_RETRIEVER={comum.RETRIEVER}.")
    else:
        print("\n(nada foi gravado. Use --gravar com|sem para produzir os "
              "embeddings das perguntas.)")


if __name__ == "__main__":
    main()
