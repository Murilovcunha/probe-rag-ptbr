"""
04_metricas.py -- Consolidação e análise sem GPU.

Roda sobre os parquet devolvidos pelo colega. Além de consolidar e calcular o
BERTScore, implementa as análises §E/§F da qualificação da rodada:

  16. junta o Bloco 1 com o gabarito das 74 (join, não rodada nova)
  18/23. nula EXATA da concordância D-entropia e de D_embaralhado
  19. curva dose-resposta da posição do gold, analítica
  20. BERTScore das respostas por posição -> rótulo "chunk suficiente" [PRINCIPAL]
  21. posto do gold dentro da pergunta, sob os 4 escores           [PRINCIPAL]
  22. o baseline de similaridade é idêntico ao Precision@1         [REENQUADRA]

LEITURA CENTRAL (item 22). `max_similaridade_e_gold` não é um baseline com
variância própria: é uma identidade com o Precision@1 do retriever, que é o
mesmo para os quatro modelos por desenho. Por isso `min_entropia_e_gold`
passou a métrica SECUNDÁRIA, e a comparação principal do Bloco 2 é o posto do
gold (item 21) sobre um rótulo não-degenerado (item 20).

DUAS MODALIDADES NOS DOIS BLOCOS. Desde a rodada de 2026-09-05 o Bloco 2
também roda sob os dois critérios de parada da seção 4.3.2 da tese --
"regex" (Modalidade A, Eq. 21) e "json" (Modalidade B, Eq. 22). Toda análise
do Bloco 2 é, portanto, estratificada por `criterio`, e a comparação pareada
A x B aparece ao final de cada seção. Arquivos antigos, sem a coluna, são
lidos como `criterio = "regex"`.

RESSALVA DA MODALIDADE B. Sob o contrato JSON os primeiros tokens gerados são
`{"resposta": "` -- sintaxe, não resposta. O gerador desloca a janela de
entropia para depois da abertura da chave (comum.ALINHAR_JANELA_ENTROPIA_JSON)
e registra em `janela_json_alinhada` se conseguiu. Onde não conseguiu, a
entropia mede sintaxe e NÃO é comparável com a Modalidade A: essas posições
são descartadas das análises de entropia (item 06, 21, circularidade) e o
descarte é reportado. Elas continuam valendo para D, conformidade e formato.

Uso:
  python 04_metricas.py
  python 04_metricas.py --recalcular         # refaz o BERTScore de tudo
  python 04_metricas.py --sem-bertscore-740  # pula o item 20 (é o passo lento)
  python 04_metricas.py --manter-janela-json-desalinhada
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

import comum

# Item 20: o limiar de "chunk suficiente" é um QUANTIL das respostas geradas
# sobre os chunks-ouro anotados, não um valor absoluto de BERTScore. Sem
# rescale_with_baseline o BERT português comprime tudo em ~0,6-0,9 e um corte
# absoluto seria arbitrário. Quantil é pré-declarável e sobrevive à checagem
# de sensibilidade (que é reportada logo abaixo dele).
QUANTIL_SUFICIENTE = 0.10
QUANTIS_SENSIBILIDADE = (0.05, 0.10, 0.25)

# Item 21: com 12 das 74 perguntas tendo mais de um chunk-ouro, "posto do
# gold" precisa de regra declarada. Adotada: MELHOR posto entre os golds.
REGRA_POSTO_MULTI_GOLD = "melhor"

# Modalidade B: descartar das análises de ENTROPIA as posições em que a janela
# não pôde ser deslocada para o valor da chave "resposta" (o modelo quebrou o
# formato). Nessas a entropia mede a sintaxe do JSON. Desligável pela CLI para
# quem quiser ver o efeito do filtro.
FILTRAR_JANELA_JSON_DESALINHADA = True

# Escores comparados no item 21. sinal=+1 -> valor MAIOR é melhor.
ESCORES = [
    ("entropia_media", -1, "entropia (menor = mais confiante)"),
    ("similaridade", +1, "similaridade de cosseno (retriever)"),
    ("logprob_medio_referencia", +1, "teacher forcing do gabarito (item 13)"),
    ("sonda_p_sim_norm", +1, "sonda sim/não (item 14)"),
    ("sobreposicao_lexical", +1, "sobreposição lexical (controle do item 13)"),
]


# ============================================================
# utilidades
# ============================================================

def _ler(padrao):
    arqs = sorted(glob.glob(os.path.join(comum.DIR_SAIDAS, padrao)))
    arqs = [a for a in arqs if "_parcial" not in a and "_checkpoint" not in a]
    if not arqs:
        arqs = sorted(glob.glob(os.path.join(
            comum.DIR_SAIDAS, padrao.replace(".parquet", "_checkpoint_sem_bertscore.parquet"))))
    if not arqs:
        # última linha de defesa: rodada interrompida, só o parcial existe
        arqs = sorted(glob.glob(os.path.join(
            comum.DIR_SAIDAS, padrao.replace(".parquet", "_parcial.parquet"))))
        if arqs:
            print(f"  [aviso] usando arquivos PARCIAIS ({len(arqs)}): a rodada "
                  f"correspondente não terminou. Os números são de uma amostra.")
    if not arqs:
        return None
    return pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)


def _talvez_bertscore(df, col_gerada, col_ref, recalcular):
    if not len(df):
        return df
    if "S_bertscore_f1" in df.columns and not recalcular:
        return df
    print(f"  calculando BERTScore de {len(df)} respostas...")
    try:
        P, R, F1 = comum.calcular_bertscore(df[col_gerada].tolist(), df[col_ref].tolist())
        df["S_bertscore_p"], df["S_bertscore_r"], df["S_bertscore_f1"] = P, R, F1
    except Exception as e:  # noqa: BLE001
        print(f"  [aviso] BERTScore indisponível ({type(e).__name__}: {e}).")
        print("  As colunas S_bertscore_* ficarão ausentes; o resto é calculado normalmente.")
    return df


def _fmt(x):
    return f"{x:,.3f}"


# ============================================================
# BLOCO 1
# ============================================================

def _cobertura(df, rotulo):
    """
    Matriz modelo x retriever. Existe porque a comparação ENTRE MODELOS só é
    válida dentro de um mesmo retriever: se o modelo A rodou só com qwen3 e o
    B só com e5small, a diferença entre eles mistura modelo com recuperação.
    O 04 estratifica e portanto nunca soma os dois -- mas um buraco na matriz
    faz a tabela sair com um modelo sozinho em cada bloco, e isso passa
    despercebido quem não estiver procurando.
    """
    if "retriever" not in df.columns or df["retriever"].nunique() < 2:
        return
    m = df.pivot_table(index="modelo", columns="retriever",
                       values="pergunta_idx", aggfunc="nunique").fillna(0).astype(int)
    print(f"\n  cobertura modelo x retriever ({rotulo}), em perguntas:")
    print(m.to_string())
    faltando = (m == 0).sum().sum()
    completos = int((m > 0).all(axis=1).sum())
    if faltando:
        print(f"  <<< {faltando} célula(s) vazia(s). Só {completos} modelo(s) rodaram")
        print("      nos DOIS retrievers -- só esses sustentam a comparação 2x2.")
        print("      Os demais comparam-se apenas dentro do próprio retriever.")
    else:
        print("  OK: todos os modelos rodaram nos dois retrievers.")


def consolidar_bloco1(recalcular):
    df = _ler("b1_resumo_*.parquet")
    if df is None:
        print("Bloco 1: nada encontrado em saidas/.")
        return None

    print(f"\n{'=' * 78}\n### BLOCO 1\n{'=' * 78}")
    df = _talvez_bertscore(df, "resposta_final", "resposta_gabarito", recalcular)
    if "criterio" not in df.columns:      # saídas anteriores à Modalidade B
        df["criterio"] = "regex"
    # O glob pega TODOS os retrievers de uma vez. Isso é proposital -- é a
    # comparação 2x2 -- mas só funciona se o retriever for chave de
    # agrupamento. Poolar qwen3 com e5small daria uma média sem sentido.
    if "retriever" not in df.columns:     # saídas anteriores a 2026-09-06
        df["retriever"] = "qwen3"
    if df["retriever"].nunique() > 1:
        print(f"  retrievers presentes: {sorted(df['retriever'].unique())} "
              f"-- todas as tabelas abaixo são estratificadas por ele")
        _cobertura(df, "Bloco 1")

    # ---- item 16: join com o gabarito das 74 -------------------------
    # As 74 perguntas anotadas são as primeiras 74 das 192, em ordem contígua
    # (verificado: pergunta_idx 0..73). O Bloco 1 JÁ rodou sobre elas, então
    # isto é um join, não uma rodada nova. Acrescenta o único ponto em que D é
    # medido no RANKING REAL com chunk-ouro conhecido.
    if os.path.exists(comum.ARQ_PERGUNTAS_74):
        g = pd.read_parquet(comum.ARQ_PERGUNTAS_74)[
            ["pergunta_idx", "id_chunk_gold", "ids_chunk_gold_todos"]]
        df = df.merge(g, on="pergunta_idx", how="left")
        tem_gold = df["ids_chunk_gold_todos"].notna()
        df["parou_no_gold"] = [
            (row["id_chunk_parada"] is not None
             and not pd.isna(row["id_chunk_parada"])
             and int(row["id_chunk_parada"]) in {int(x) for x in row["ids_chunk_gold_todos"]})
            if row["ids_chunk_gold_todos"] is not None
            and not np.isscalar(row["ids_chunk_gold_todos"]) else None
            for _, row in df.iterrows()
        ]
        print(f"  item 16: {int(tem_gold.sum())} linhas com chunk-ouro conhecido "
              f"({df.loc[tem_gold, 'pergunta_idx'].nunique()} perguntas das 192)")

    # ---- tabela por modelo x critério x modo -------------------------
    linhas = []
    for (retriever, modelo, criterio, modo), sub in df.groupby(
            ["retriever", "modelo", "criterio", "modo"]):
        parou = sub[sub["D_iter"].notna()]
        n_infer = sub["n_inferencias"].sum()
        linha = {
            "retriever": retriever, "modelo": modelo, "criterio": criterio,
            "modo": modo, "n": len(sub),
            "parou_%": 100 * len(parou) / len(sub),
            "censurado_dmax": int(sub["censurado_direita"].sum()),
            "censurado_tokens": int(sub["censurado_limite_tokens"].sum()),
            # médias de D só sobre as NÃO censuradas (tese, 4.4.1)
            "D_iter_medio": parou["D_iter"].mean() if len(parou) else np.nan,
            "D_iter_mediana": parou["D_iter"].median() if len(parou) else np.nan,
            "D_chunk_medio": parou["D_chunk"].mean() if len(parou) else np.nan,
            "T_total_medio": sub["tokens_total_acum"].mean(),
            "n_inferencias_media": sub["n_inferencias"].mean(),
            "conformidade_json_%": (100 * sub["n_iter_conformes"].sum() / n_infer
                                    if criterio == "json" and n_infer else np.nan),
            "json_estrito_%": (100 * sub["n_iter_json_estrito"].sum() / n_infer
                               if criterio == "json" and n_infer else np.nan),
            "max_tokens_%": (100 * sub["n_iter_max_tokens"].sum() / n_infer
                             if "n_iter_max_tokens" in sub and n_infer else np.nan),
            # P e R ao lado do F1, e o comprimento: sem isso o F1 esconde o
            # confundidor. Medido em 09/09/2026 nos três modelos: o braço
            # iterativo tem precisão MAIOR e recall MENOR que o fixo50 em
            # quase todas as células, e o sinal do F1 é decidido por onde cada
            # braço cai na curva comprimento-resposta / comprimento-gabarito.
            # Comparar F1 entre braços de comprimento sistematicamente
            # diferente mede comprimento, não qualidade.
            "S_f1_medio": sub["S_bertscore_f1"].mean() if "S_bertscore_f1" in sub else np.nan,
            "S_p_medio": sub["S_bertscore_p"].mean() if "S_bertscore_p" in sub else np.nan,
            "S_r_medio": sub["S_bertscore_r"].mean() if "S_bertscore_r" in sub else np.nan,
            "razao_compr_mediana": (
                (sub["resposta_final"].astype(str).str.split().str.len()
                 / sub["resposta_gabarito"].astype(str).str.split().str.len()
                   .replace(0, np.nan)).median()
                if {"resposta_final", "resposta_gabarito"} <= set(sub.columns) else np.nan),
            "tempo_medio_s": sub["tempo_s"].mean(),
            "vram_pico_gb": sub["vram_pico_gb"].max(),
        }
        if "parou_no_gold" in sub.columns:
            comgold = sub[sub["parou_no_gold"].notna()]
            linha["parou_no_gold_%"] = (100 * comgold["parou_no_gold"].mean()
                                        if len(comgold) else np.nan)
        linhas.append(linha)
    tabela = pd.DataFrame(linhas).sort_values(
        ["retriever", "modelo", "criterio", "modo"])
    print(tabela.to_string(index=False, float_format=_fmt))

    # ---- truncamento da RESPOSTA: contamina o eixo S -----------------
    # O teto de max_new_tokens não é neutro entre os braços: o fixo50 manda
    # 50 páginas e o modelo tenta resumir tudo, batendo no teto muito mais que
    # os braços com parada. Truncar derruba o BERTScore, então um teto baixo
    # enviesa o eixo S CONTRA o baseline -- na direção que favorece a tese.
    if "n_iter_max_tokens" in df.columns:
        tr = df.groupby(["retriever", "modelo", "criterio", "modo"]).apply(
            lambda s: pd.Series({
                "trunc_%": 100 * s["n_iter_max_tokens"].sum() / max(s["n_inferencias"].sum(), 1),
                "S_truncadas": s.loc[s["n_iter_max_tokens"] > 0, "S_bertscore_f1"].mean()
                if "S_bertscore_f1" in s else np.nan,
                "S_inteiras": s.loc[s["n_iter_max_tokens"] == 0, "S_bertscore_f1"].mean()
                if "S_bertscore_f1" in s else np.nan,
            }), include_groups=False)
        print("\n  Truncamento da resposta (teto de max_new_tokens):")
        print(tr.to_string(float_format=_fmt))
        piv = tr["trunc_%"].unstack("modo")
        if "fixo50" in piv.columns:
            outros = [c for c in ("iterativo", "incremental") if c in piv.columns]
            if outros:
                folga = (piv["fixo50"] - piv[outros].max(axis=1)).max()
                if folga > 5:
                    print(f"  <<< ATENÇÃO: o fixo50 trunca {folga:.0f} pontos percentuais MAIS")
                    print("      que os braços com parada. O eixo S do baseline está")
                    print("      subestimado por artefato de teto, não por qualidade.")
                    print("      Compare S só entre respostas NÃO truncadas, ou suba")
                    print("      LOGPROB_MAX_NEW_TOKENS e refaça. Ver README 9.32.")

    # ---- alerta de censura por limite de tokens ----------------------
    inc = df[df["modo"] == "incremental"]
    if len(inc):
        taxa = 100 * inc["censurado_limite_tokens"].mean()
        print(f"\n  censurado_limite_tokens (incremental): {taxa:.1f}%")
        if taxa > 5:
            print("  <<< ATENÇÃO: o braço incremental foi interrompido por limite de VRAM,")
            print("      não por ancoragem. Qualquer comparação iterativo x incremental")
            print("      precisa reportar esta taxa ao lado.")

    # ---- Modalidade A x Modalidade B, pareado ------------------------
    if df["criterio"].nunique() > 1:
        print("\n  Modalidade A (regex) x Modalidade B (JSON), pareado por pergunta:")
        comp = []
        for (modelo, modo), sub in df.groupby(["modelo", "modo"]):
            piv_d = sub.pivot_table(index="pergunta_idx", columns="criterio", values="D_iter")
            piv_t = sub.pivot_table(index="pergunta_idx", columns="criterio",
                                    values="tokens_total_acum")
            if not {"regex", "json"} <= set(piv_d.columns):
                continue
            ambos = piv_d.dropna()
            linha = {
                "modelo": modelo, "modo": modo, "n_pareado": len(ambos),
                "D_regex_medio": ambos["regex"].mean() if len(ambos) else np.nan,
                "D_json_medio": ambos["json"].mean() if len(ambos) else np.nan,
                "delta_D": (ambos["json"] - ambos["regex"]).mean() if len(ambos) else np.nan,
                "D_igual_%": (100 * (ambos["json"] == ambos["regex"]).mean()
                              if len(ambos) else np.nan),
                "delta_T_%": (100 * (piv_t["json"].mean() / piv_t["regex"].mean() - 1)
                              if piv_t["regex"].mean() else np.nan),
            }
            if "S_bertscore_f1" in sub.columns:
                piv_s = sub.pivot_table(index="pergunta_idx", columns="criterio",
                                        values="S_bertscore_f1")
                linha["delta_S"] = (piv_s["json"] - piv_s["regex"]).mean()
            comp.append(linha)
        if comp:
            print(pd.DataFrame(comp).to_string(index=False, float_format=_fmt))
            print("  'n_pareado' conta só perguntas em que AS DUAS modalidades pararam;")
            print("  censuras assimétricas estão em censurado_dmax e vão à parte.")

        js = df[df["criterio"] == "json"]
        if len(js) and "D_iter_regra_alternativa" in js.columns:
            div = (js["D_iter"].fillna(-1) != js["D_iter_regra_alternativa"].fillna(-1))
            print(f"\n  Modalidade B -- D difere sob 'sentinela OU regex no valor' em "
                  f"{int(div.sum())}/{len(js)} ({100 * div.mean():.1f}%).")
            print("  D_iter é a Eq. 22; D_iter_regra_alternativa é a leitura robusta.")


    # ---- iterativo x incremental: CONTAGEM POR DIREÇÃO ----------------
    # A média de (D_inc - D_iter) é o resumo errado para este contraste.
    # Os dois pilotos sob e5 mostraram os dois mecanismos, em perguntas
    # diferentes e em sentidos OPOSTOS:
    #   q136: iterativo parou em D=5, incremental nunca parou  -> acumular
    #         DILUIU o contexto e piorou o reconhecimento
    #   q174: incremental parou em D=3, iterativo nunca parou  -> a resposta
    #         exigia COMPOR mais de um chunk, e só acumulando dava
    # Se os dois efeitos tiverem massa parecida, a média cancela e o
    # resultado parece "sem efeito" quando há dois efeitos reais brigando.
    # Por isso a leitura primária deste contraste é a CONTAGEM por direção.
    print("\n  iterativo x incremental, por DIREÇÃO (não por média):")
    dirs = []
    for (retriever, modelo, criterio), sub in df.groupby(["retriever", "modelo", "criterio"]):
        piv = sub[sub["modo"].isin(["iterativo", "incremental"])].pivot_table(
            index="pergunta_idx", columns="modo",
            values=["D_iter", "censurado_direita"], aggfunc="first")
        if ("D_iter", "iterativo") not in piv.columns:
            continue
        di, dn = piv[("D_iter", "iterativo")], piv[("D_iter", "incremental")]
        ci = piv.get(("censurado_direita", "iterativo"))
        cn = piv.get(("censurado_direita", "incremental"))
        cont = {"retriever": retriever, "modelo": modelo, "criterio": criterio,
                "n": len(piv)}
        ambos = di.notna() & dn.notna()
        cont["igual"] = int((ambos & (di == dn)).sum())
        cont["incremental_antes"] = int((ambos & (dn < di)).sum())
        cont["iterativo_antes"] = int((ambos & (dn > di)).sum())
        cont["so_incremental_parou"] = int((di.isna() & dn.notna()).sum())
        cont["so_iterativo_parou"] = int((dn.isna() & di.notna()).sum())
        cont["ambos_censurados"] = int((di.isna() & dn.isna()).sum())
        del ci, cn
        dirs.append(cont)
    if dirs:
        dd = pd.DataFrame(dirs)
        print(dd.to_string(index=False))
        print("  `so_incremental_parou`: acumular RESOLVEU o que uma página só não")
        print("     resolvia -- resposta que exige compor mais de um chunk.")
        print("  `so_iterativo_parou`  : acumular ATRAPALHOU -- o modelo deixou de")
        print("     reconhecer a suficiência que já tinha visto (diluição).")
        print("  Os dois são resultado. Se aparecerem em números parecidos, a média")
        print("  de delta_D vai dar ~0 e ESCONDER os dois; reporte as contagens.")
        tot_i = int(dd["so_incremental_parou"].sum())
        tot_t = int(dd["so_iterativo_parou"].sum())
        if tot_i and tot_t:
            print(f"  <<< ATENÇÃO: os dois sentidos ocorrem ({tot_i} e {tot_t}). A média")
            print("      de delta_D NÃO resume este contraste -- use as contagens e")
            print("      caracterize as perguntas de cada lado.")
        dd.to_csv(os.path.join(comum.DIR_SAIDAS,
                               "direcao_iter_vs_incremental.csv"), index=False)

    # ---- eixo Z ------------------------------------------------------
    etas = []
    for (retriever, modelo, criterio), sub in df.groupby(
            ["retriever", "modelo", "criterio"]):
        piv = sub.pivot_table(index="pergunta_idx", columns="modo", values="tokens_total_acum")
        if "fixo50" not in piv.columns:
            continue
        for modo in ("iterativo", "incremental"):
            if modo in piv.columns:
                eta = 1 - piv[modo] / piv["fixo50"]
                # A MÉDIA de eta é um resumo ruim: a distribuição é de cauda
                # longa à esquerda. Medido em 09/09/2026: no incremental do
                # qwen3.8 a média é -0,73 e a MEDIANA +0,96, com eta >= 0,9 em
                # 70% das perguntas. Dizer "gasta mais em média" sugere que o
                # método nunca compensa, quando ele compensa na maioria e
                # explode numa minoria -- que é outra afirmação, e a útil.
                etas.append({"retriever": retriever, "modelo": modelo,
                             "criterio": criterio, "modo": modo,
                             "eta_medio": eta.mean(), "eta_mediana": eta.median(),
                             "eta_p10": eta.quantile(0.10),
                             "eta_maior_0.9_%": 100 * (eta >= 0.9).mean(),
                             "eta_negativo_%": 100 * (eta < 0).mean(),
                             "T_fix_medio": piv["fixo50"].mean()})
    if etas:
        print("\n  Eixo Z -- eta = 1 - T/T_fix(k=50), pareado por pergunta:")
        print(pd.DataFrame(etas).to_string(index=False, float_format=_fmt))

    # ---- similaridade na parada (item 10) ----------------------------
    if "similaridade_na_parada" in df.columns:
        s = df[df["similaridade_na_parada"].notna()]
        if len(s):
            print("\n  Similaridade do chunk em que o método parou (item 10):")
            print(s.groupby(["modelo", "modo"])["similaridade_na_parada"]
                  .agg(["mean", "median", "min", "count"]).to_string(float_format=_fmt))

    destino = os.path.join(comum.DIR_SAIDAS, "consolidado_bloco1.parquet")
    df.to_parquet(destino, index=False)
    df.to_csv(destino.replace(".parquet", ".csv"), index=False)
    tabela.to_csv(os.path.join(comum.DIR_SAIDAS, "tabela_bloco1.csv"), index=False)
    print(f"\n  -> {destino}")
    return df


# ============================================================
# BLOCO 2 -- itens 18 a 23
# ============================================================

def _nula_exata(g, v, K=comum.K_TOP):
    """
    Item 18. No Bloco 2 cada prompt contém UM chunk, então a entropia não
    depende da permutação: a concordância é o evento "o chunk de menor
    entropia é o primeiro válido em pi", com pi sorteada independentemente de
    tudo o que foi medido. Condicionando nos dados da pergunta:

        E[concordância | dados] = 1{argmin é válido} / V
        E[D_embaralhado | V]    = (K + 1) / (V + 1)

    `g` = o chunk de menor entropia é válido?   `V` = nº de posições válidas.
    V = 0 -> censurado (nada a esperar).
    """
    if not v:
        return None, None
    return (1.0 / v if g else 0.0), (K + 1) / (v + 1)


def _dose_resposta(g_valido, m_validos_nao_gold, K=comum.K_TOP):
    """
    Item 19. Curva dose-resposta da posição do gold, ANALÍTICA.

    Colocar o gold em cada posição j = 1..K parecia custar 10x mais geração.
    Como a posição não entra no prompt, é reetiquetagem das mesmas 10
    gerações. Com o gold fixo em j e os outros K-1 chunks distribuídos
    uniformemente nas posições restantes, o método para NO gold se e só se o
    gold é válido e nenhum outro válido caiu antes de j:

        P(parar no gold | gold em j) = 1{gold válido} * C(K-j, m) / C(K-1, m)

    onde m = nº de válidos entre os outros K-1. Também devolvemos E[D | j].
    """
    n_out = K - 1
    if m_validos_nao_gold > n_out:
        return None
    p_stop, e_d = [], []
    for j in range(1, K + 1):
        # posições depois de j entre as não-j: K - j
        num = math.comb(K - j, m_validos_nao_gold) if K - j >= m_validos_nao_gold else 0
        den = math.comb(n_out, m_validos_nao_gold)
        p = (num / den) if den else 0.0
        p_stop.append(p if g_valido else 0.0)

        # E[D | gold em j] = soma_{t=0}^{K-1} P(D > t)
        esperado = 0.0
        for t in range(0, K):
            a_t = (K - t) - (1 if j > t else 0)   # não-j com posição > t
            if a_t < m_validos_nao_gold:
                p_sem = 0.0
            else:
                p_sem = math.comb(a_t, m_validos_nao_gold) / den if den else 0.0
            p_maior = p_sem * (0.0 if (g_valido and j <= t) else 1.0)
            esperado += p_maior
        e_d.append(esperado)
    return p_stop, e_d


def _posto(valores, is_gold, sinal, regra=REGRA_POSTO_MULTI_GOLD):
    """
    Item 21. Posto do gold entre as 10 posições sob um escore.

    Por que esta é a estatística certa: é uma medida POR PERGUNTA (n = 74
    independentes), o que permite teste pareado direto entre os escores e
    dispensa o cluster bootstrap que um pool de 740 pares aninhados exigiria;
    e é invariante a reescala monotônica, o que torna a comparação entre
    modelos válida sem normalizar nats.

    Multi-gold (12 das 74 perguntas): regra declarada = MELHOR posto.
    """
    v = np.asarray(valores, dtype=float)
    g = np.asarray(is_gold, dtype=bool)
    if not g.any() or np.all(np.isnan(v)):
        return None
    ordem = np.argsort(-sinal * v, kind="stable")   # melhor primeiro
    postos = np.empty(len(v), dtype=int)
    postos[ordem] = np.arange(1, len(v) + 1)
    p_gold = postos[g]
    return int(p_gold.min()) if regra == "melhor" else float(p_gold.mean())


def consolidar_bloco2(recalcular, com_bertscore_740=True,
                      filtrar_janela_json=FILTRAR_JANELA_JSON_DESALINHADA):
    df = _ler("b2_resultados_*.parquet")
    ent = _ler("b2_entropia_*.parquet")
    if df is None:
        print("Bloco 2: nada encontrado em saidas/.")
        return None

    print(f"\n{'=' * 78}\n### BLOCO 2\n{'=' * 78}")
    df = _talvez_bertscore(df, "texto_resposta_final", "resposta_gabarito", recalcular)

    # ---- modalidades presentes ---------------------------------------
    # Saídas anteriores a 2026-09-05 não têm a coluna: são regex por definição.
    if "criterio" not in df.columns:
        df["criterio"] = "regex"
    if ent is not None and len(ent) and "criterio" not in ent.columns:
        ent["criterio"] = "regex"
    if "retriever" not in df.columns:
        df["retriever"] = "qwen3"
    if ent is not None and len(ent) and "retriever" not in ent.columns:
        ent["retriever"] = "qwen3"
    criterios = sorted(df["criterio"].dropna().unique().tolist())
    # retriever antes de modelo: com dois retrievers, a leitura natural é
    # "dentro deste retriever, como os modelos se comportam?"
    GRP = ["retriever", "modelo", "criterio"]
    if df["retriever"].nunique() > 1:
        print(f"  retrievers presentes: {sorted(df['retriever'].unique())}")
        _cobertura(df, "Bloco 2")
    print(f"\n  modalidades de parada presentes: {criterios}")
    if len(criterios) == 1:
        print("  (só uma -- a comparação pareada A x B fica de fora desta rodada)")

    # `uma` = uma linha por (modelo, pergunta): tudo que descreve o RANKING é
    # idêntico nas duas modalidades, porque o sorteio e os embeddings são os
    # mesmos. Usar df inteiro aqui contaria cada pergunta duas vezes.
    uma = df.drop_duplicates(["retriever", "modelo", "pergunta_idx"])

    # ---- item 22a: verificação de sanidade grátis --------------------
    print("\n--- item 22a: caracterização do RETRIEVER (invariante ao modelo) ---")
    print("  max_similaridade_e_gold é idêntico ao Precision@1 do retriever e")
    print("  depende só do ranking, que é o mesmo para todos os modelos por")
    print("  desenho (embeddings congelados). As linhas abaixo TÊM que bater.")
    ret = uma.groupby(["retriever", "modelo"]).agg(
        max_simil_e_gold_pct=("max_similaridade_e_gold", lambda s: 100 * s.mean()),
        n_golds_no_top10=("n_golds_no_top10", "mean"),
        reposicionado_pct=("gold_foi_reposicionado_no_top10", lambda s: 100 * s.mean()),
        gold_em_1o_pct=("posicao_gold_no_ranking_natural", lambda s: 100 * (s == 1).mean()),
    )
    print(ret.to_string(float_format=_fmt))
    if ret.nunique().max() > 1:
        print("  <<< ATENÇÃO: as linhas divergem. Algo quebrou no ranking ou nos")
        print("      embeddings congelados -- investigue ANTES de usar os números.")
    else:
        print("  OK: idênticas, como esperado.")
    checa = uma.groupby(["retriever", "modelo"]).apply(
        lambda s: float((s["max_similaridade_e_gold"].astype(bool)
                         == (s["posicao_gold_no_ranking_natural"] == 1)).mean()),
        include_groups=False)
    print(f"  identidade max_simil_e_gold == (gold em 1º): "
          f"{100 * float(checa.mean()):.1f}% das linhas")
    if len(criterios) > 1:
        inv = df.groupby(["retriever", "modelo", "pergunta_idx"])[
            "max_similaridade_e_gold"].nunique()
        n_div = int((inv > 1).sum())
        print(f"  invariância ao critério (mesma pergunta, regex x json): "
              f"{n_div} divergência(s) em {len(inv)} perguntas"
              f"{'  <<< ERRO: o ranking não deveria mudar' if n_div else '  OK'}")

    # ---- item 22b/c: as DUAS perguntas -------------------------------
    # Correção sobre a v1 da qualificação: o estrato degenerado é
    # `gold_foi_reposicionado`, NÃO "o retrieval não acertou de primeira".
    # Com o gold em rank 2..10 ele está no conjunto com a similaridade REAL
    # dele, então sob a estatística de POSTO (item 21) a similaridade continua
    # informativa. Só no caso reposicionado o desenho força o gold a ser o de
    # menor similaridade dos dez, e aí o posto dele é 10 por construção.
    df["estrato"] = np.where(
        df["gold_foi_reposicionado_no_top10"], "B_reposicionado", "A_top10_natural")
    uma = df.drop_duplicates(["retriever", "modelo", "pergunta_idx"])   # com a coluna estrato
    print("\n--- item 22c: estratos, com a correção do corte ---")
    print(uma.groupby(["retriever", "modelo", "estrato"]).size().rename("n").to_string())
    print("  A_top10_natural : o gold está no top-10 com a similaridade real dele.")
    print("                    Comparador legítimo -> comparação principal (item 21).")
    print("  B_reposicionado : o desenho inseriu o gold como o MENOR dos dez, logo o")
    print("                    posto da similaridade é 10 por construção. Aqui não há")
    print("                    comparação possível: a pergunta vira 'a entropia recupera")
    print("                    o chunk que o retriever não trouxe?', contra o acaso.")

    # ---- itens 18 e 23: nula exata -----------------------------------
    nulas = []
    for _, r in df.iterrows():
        v = int(r["n_posicoes_validas"])
        # o chunk de menor entropia é válido?
        g = bool(r["min_entropia_e_gold"]) if v else False
        e_conc, e_d = _nula_exata(g, v)
        nulas.append({"retriever": r["retriever"], "modelo": r["modelo"],
                      "criterio": r["criterio"],
                      "pergunta_idx": r["pergunta_idx"],
                      "V": v, "E_concordancia_nula": e_conc, "E_D_nula": e_d,
                      "concordancia_obs": bool(r["concordancia_D_entropia"]),
                      "D_obs": r["D_embaralhado"]})
    nul = pd.DataFrame(nulas)
    print("\n--- itens 18 e 23: nula EXATA (sem Monte Carlo), por modalidade ---")
    tab_nul = nul.groupby(GRP).agg(
        concordancia_obs_pct=("concordancia_obs", lambda s: 100 * s.mean()),
        concordancia_nula_pct=("E_concordancia_nula", lambda s: 100 * s.mean()),
        D_obs_medio=("D_obs", "mean"),
        D_nula_medio=("E_D_nula", "mean"),
        V_medio=("V", "mean"),
        censurado_V0=("V", lambda s: int((s == 0).sum())),
    )
    print(tab_nul.to_string(float_format=_fmt))
    print("  A concordância só é informativa se superar a coluna 'nula'.")
    print("  As quantidades invariantes ao sorteio são P(argmin válido) e a")
    print("  distribuição de V, não a concordância em si.")
    print("  A nula muda entre as modalidades porque V muda: cada critério de")
    print("  parada declara um conjunto diferente de posições como válidas.")

    tabelas_extra = {"nula_exata_bloco2": tab_nul.reset_index()}
    p_estrat = None      # preenchido pelo item 21b, usado na tabela final

    # ============================================================
    # análises que precisam do arquivo longo (uma linha por posição)
    # ============================================================
    if ent is not None and len(ent):
        # ---- conformidade e alinhamento da Modalidade B -------------
        # Esta seção vem ANTES de tudo o que usa entropia, porque decide
        # quais linhas da Modalidade B são comparáveis com a Modalidade A.
        if "json" in criterios and "janela_json_alinhada" in ent.columns:
            js = ent[ent["criterio"] == "json"]
            print("\n--- Modalidade B: formato e janela de entropia ---")
            tab_js = js.groupby(["retriever", "modelo"]).agg(
                n=("janela_json_alinhada", "size"),
                janela_alinhada_pct=("janela_json_alinhada",
                                     lambda s: 100 * (s == True).mean()),  # noqa: E712
                conformidade_pct=("conformidade_json",
                                  lambda s: 100 * (s == True).mean()),     # noqa: E712
                json_estrito_pct=("json_estrito",
                                  lambda s: 100 * (s == True).mean()),     # noqa: E712
            )
            print(tab_js.to_string(float_format=_fmt))
            print("  janela_alinhada = a janela de 20 tokens pôde ser deslocada para")
            print("  DEPOIS de {\"resposta\": \" e mede a resposta, não a sintaxe.")
            print("  Abaixo de ~80% a entropia da Modalidade B daquele modelo não é")
            print("  comparável com a da Modalidade A nem depois do filtro (o que")
            print("  sobra é um subconjunto selecionado pelo próprio modelo).")
            tabelas_extra["modalidade_b_formato"] = tab_js.reset_index()

        # ---- item 04: regra de agregação ----------------------------
        print(f"\n--- item 04: n_passos_entropia (mínimo para agregar = "
              f"{comum.MIN_PASSOS_ENTROPIA}) ---")
        print(ent.groupby(GRP)["n_passos_entropia"]
              .agg(["mean", "median", "min", "max"]).to_string(float_format=_fmt))
        curtos = ent["n_passos_entropia"] < comum.MIN_PASSOS_ENTROPIA
        print(f"  posições descartadas por janela curta: {int(curtos.sum())}"
              f"/{len(ent)} ({100 * curtos.mean():.1f}%)")

        # filtro da Modalidade B: fora as posições em que a janela ficou na
        # sintaxe. `== False` de propósito: None (Modalidade A) fica.
        desalinhadas = pd.Series(False, index=ent.index)
        if filtrar_janela_json and "janela_json_alinhada" in ent.columns:
            desalinhadas = ((ent["criterio"] == "json")
                            & (ent["janela_json_alinhada"] == False))  # noqa: E712
            if desalinhadas.any():
                print(f"  posições da Modalidade B descartadas por janela na sintaxe: "
                      f"{int(desalinhadas.sum())}"
                      f"/{int((ent['criterio'] == 'json').sum())} das linhas json")
                print("  (--manter-janela-json-desalinhada mantém-nas, para ver o efeito)")

        ent_ok = ent[~curtos & ~desalinhadas].copy()

        # ---- item 03: auditoria do alinhamento ----------------------
        if "inicio_janela_entropia" in ent.columns:
            print("\n--- item 03: fração de posições com inicio > 0 ---")
            fr = ent.groupby(GRP)["inicio_janela_entropia"].apply(lambda s: (s > 0).mean())
            print((100 * fr).to_string(float_format=_fmt))
            print("  Se um modelo com marcadores declarados vier 0%, o alinhamento não")
            print("  aconteceu e a entropia mediu o canal de análise -- número inválido.")
            print("  Na Modalidade B o deslocamento também pula a abertura da chave,")
            print("  então ali inicio > 0 é o caso NORMAL, não a exceção.")

        # ---- §G: invariante de índice de logit ----------------------
        if "n_divergencias_argmax" in ent.columns:
            d = ent.groupby(GRP)[["n_divergencias_argmax", "n_passos_verificados"]].sum()
            d["taxa_%"] = 100 * d["n_divergencias_argmax"] / d["n_passos_verificados"]
            print("\n--- §G: argmax(logit_i) == token gerado_i (greedy) ---")
            print(d.to_string(float_format=_fmt))
            print("  Taxa > 0 significa que um logits processor interferiu "
                  "(repetition_penalty / min_new_tokens).")

        # ---- item 06: entropia por janela ---------------------------
        print("\n--- item 06: janela primária (20) x janelas mais largas ---")
        for n in (5, 10, 20, 50, 100):
            ent_ok[f"H_{n}"] = ent_ok["entropia_por_token"].apply(
                lambda v, n=n: float(np.mean(v[:n])) if len(v) else np.nan)
        ent_ok["H_todos"] = ent_ok["entropia_media_todos"]
        cols = [f"H_{n}" for n in (5, 10, 20, 50, 100)] + ["H_todos"]
        print(ent_ok.groupby(GRP + ["is_gold"])[cols].mean().to_string(float_format=_fmt))
        print("  A janela de 20 é a PRIMÁRIA, declarada a priori. As demais são")
        print("  análise secundária pré-declarada: mostram se o sinal se sustenta")
        print("  além do ponto de decisão recusa/resposta.")
        if len(criterios) > 1:
            print("  Os níveis de entropia NÃO são diretamente comparáveis entre as")
            print("  modalidades: são distribuições sobre continuações diferentes.")
            print("  O que se compara entre elas é o POSTO do gold (item 21).")

        # ---- circularidade: válida x recusa -------------------------
        print("\n--- circularidade: entropia por resposta válida x recusa ---")
        print(ent_ok.groupby(GRP + ["valida"])["entropia_media"]
              .agg(["mean", "count"]).to_string(float_format=_fmt))
        print("  Se a recusa também tiver entropia baixa, a relação é em U e o")
        print("  argmin não significa 'onde está a resposta'.")

        # o mesmo teste, com número e limiar declarado (ver comum.py 7c)
        print(f"\n  AUC 'recusa tem entropia menor que válida' (0,5 = sem "
              f"informação; alerta fora de "
              f"[{1 - comum.AUC_CIRCULARIDADE_ALERTA:.2f}; "
              f"{comum.AUC_CIRCULARIDADE_ALERTA:.2f}], nos DOIS sentidos):")
        diags, alertas = [], []
        for (grp_ret, modelo, crit), sub in ent_ok.groupby(GRP):
            d = comum.diagnostico_circularidade(sub["entropia_media"], sub["valida"])
            # com que frequência o argmin da pergunta cai numa recusa?
            arg_rec = []
            for _, q in sub.groupby("pergunta_idx"):
                q2 = q.dropna(subset=["entropia_media"])
                if len(q2):
                    arg_rec.append(not bool(q2.loc[q2["entropia_media"].idxmin(), "valida"]))
            d.update({"retriever": grp_ret, "modelo": modelo, "criterio": crit,
                      "argmin_e_recusa_%": 100 * float(np.mean(arg_rec)) if arg_rec else np.nan,
                      "n_perguntas": len(arg_rec)})
            diags.append(d)
            alertas += comum.texto_alerta_circularidade(d, f"{modelo} / {crit}")
        ddiag = pd.DataFrame(diags)[
            ["retriever", "modelo", "criterio", "auc", "sentido", "teto_recusa", "piso_valida",
             "separacao_total", "argmin_e_recusa_%", "n_recusas", "n_validas",
             "n_perguntas"]]
        print(ddiag.to_string(index=False, float_format=_fmt))
        for linha in alertas:
            print(linha)
        tabelas_extra["circularidade_bloco2"] = ddiag

        # ---- item 08: top-5 do primeiro passo -----------------------
        if "top5_tokens_primeiro_passo" in ent_ok.columns:
            def _p_recusa(row):
                toks = row["top5_tokens_primeiro_passo"]
                prbs = row["top5_probs_primeiro_passo"]
                if toks is None or len(toks) == 0:
                    return np.nan
                return float(sum(p for t, p in zip(toks, prbs)
                                 if str(t).strip().lower().startswith(("não", "nao", "n"))))
            ent_ok["p_recusa_primeiro_passo"] = ent_ok.apply(_p_recusa, axis=1)
            print("\n--- item 08: massa de recusa no primeiro passo ---")
            print(ent_ok.groupby(GRP + ["is_gold"])["p_recusa_primeiro_passo"]
                  .mean().to_string(float_format=_fmt))
            print("  Monotônica, ao contrário da entropia: separa 'confiante que")
            print("  responde' de 'confiante que recusa'.")
            if "json" in criterios:
                print("  Na Modalidade B o 'primeiro passo' é o primeiro token DEPOIS")
                print("  da abertura da chave, nas posições com janela alinhada.")

        # ---- item 20: BERTScore de todas as posições ----------------
        if com_bertscore_740 and "S_bertscore_f1_posicao" not in ent_ok.columns:
            print(f"\n--- item 20 [PRINCIPAL]: BERTScore de {len(ent)} respostas ---")
            try:
                # uma referência por (modelo, pergunta): as duas modalidades
                # respondem à mesma pergunta contra o mesmo gabarito
                refs = (df.drop_duplicates(["modelo", "pergunta_idx"])
                          .set_index(["modelo", "pergunta_idx"])["resposta_gabarito"])
                chave = list(zip(ent["modelo"], ent["pergunta_idx"]))
                ent["resposta_gabarito"] = [refs.get(k, "") for k in chave]
                P, R, F1 = comum.calcular_bertscore(
                    ent["texto_resposta"].tolist(), ent["resposta_gabarito"].tolist())
                ent["S_bertscore_f1_posicao"] = F1

                # O LIMIAR É POR MODALIDADE. `texto_resposta` já vem sem o
                # invólucro do JSON, mas o estilo da resposta sob contrato
                # estruturado é outro (mais curta, sem preâmbulo), e um limiar
                # único importaria essa diferença de estilo para dentro do
                # rótulo. Um quantil por modalidade mantém o rótulo comparável.
                ent["suficiente"] = False
                for crit in criterios:
                    m_c = ent["criterio"] == crit
                    gold = ent.loc[m_c & ent["is_gold"], "S_bertscore_f1_posicao"].dropna()
                    lim = float(gold.quantile(QUANTIL_SUFICIENTE)) if len(gold) else np.nan
                    ent.loc[m_c, "suficiente"] = (
                        ent.loc[m_c, "S_bertscore_f1_posicao"] >= lim)
                    print(f"  [{crit}] limiar (quantil {QUANTIL_SUFICIENTE:.0%} das "
                          f"respostas sobre chunks-ouro) = {lim:.4f}")
                    print(f"    'suficientes': "
                          f"{100 * ent.loc[m_c, 'suficiente'].mean():.1f}% de "
                          f"{int(m_c.sum())}  (contra "
                          f"{100 * ent.loc[m_c, 'is_gold'].mean():.1f}% rotulados gold)")
                    for q in QUANTIS_SENSIBILIDADE:
                        l2 = float(gold.quantile(q)) if len(gold) else np.nan
                        suf = 100 * (ent.loc[m_c, "S_bertscore_f1_posicao"] >= l2).mean()
                        print(f"    sensibilidade quantil {q:.0%}: limiar={l2:.4f}  "
                              f"suficientes={suf:.1f}%")

                # refaz ent_ok com as colunas novas, mantendo os dois filtros
                curtos2 = ent["n_passos_entropia"] < comum.MIN_PASSOS_ENTROPIA
                desal2 = pd.Series(False, index=ent.index)
                if filtrar_janela_json and "janela_json_alinhada" in ent.columns:
                    desal2 = ((ent["criterio"] == "json")
                              & (ent["janela_json_alinhada"] == False))  # noqa: E712
                ent_ok = ent[~curtos2 & ~desal2].copy()
                for n in (5, 10, 20, 50, 100):
                    ent_ok[f"H_{n}"] = ent_ok["entropia_por_token"].apply(
                        lambda v, n=n: float(np.mean(v[:n])) if len(v) else np.nan)
            except Exception as e:  # noqa: BLE001
                print(f"  [aviso] BERTScore por posição indisponível "
                      f"({type(e).__name__}: {e}).")

        # ---- item 21: posto do gold, por pergunta -------------------
        print("\n--- item 21 [PRINCIPAL]: posto do gold dentro da pergunta ---")
        print(f"  regra multi-gold: {REGRA_POSTO_MULTI_GOLD} posto entre os golds "
              f"(12 das 74 perguntas têm mais de um)")
        # O posto usa ent_ok, não ent: uma posição descartada (janela curta ou
        # janela na sintaxe) não tem entropia interpretável, e mantê-la no
        # conjunto ordenado contaminaria o posto do gold sob TODOS os escores.
        # Perguntas que perderam posições ficam com n_pos < 10 -- a coluna
        # registra isso para que a leitura saiba contra que acaso comparar.
        postos = []
        for (grp_ret, modelo, crit, pidx), sub in ent_ok.groupby(GRP + ["pergunta_idx"]):
            linha = {"retriever": modelo[0] if False else grp_ret,
                     "modelo": modelo, "criterio": crit, "pergunta_idx": pidx,
                     "n_pos": len(sub)}
            for col, sinal, _ in ESCORES:
                if col in sub.columns:
                    linha[f"posto_{col}"] = _posto(sub[col].to_numpy(),
                                                   sub["is_gold"].to_numpy(), sinal)
            if "suficiente" in sub.columns and sub["suficiente"].any():
                for col, sinal, _ in ESCORES:
                    if col in sub.columns:
                        linha[f"postoSUF_{col}"] = _posto(sub[col].to_numpy(),
                                                          sub["suficiente"].to_numpy(), sinal)
            postos.append(linha)
        dpost = pd.DataFrame(postos)
        cols_posto = [c for c in dpost.columns if c.startswith("posto_")]
        print("\n  posições sobreviventes por pergunta (acaso = (n_pos+1)/2):")
        print(dpost.groupby(GRP)["n_pos"].agg(["mean", "min", "count"])
              .to_string(float_format=_fmt))
        print("\n  rótulo = chunk-ouro anotado (menor posto é melhor):")
        print(dpost.groupby(GRP)[cols_posto].mean().to_string(float_format=_fmt))
        cols_suf = [c for c in dpost.columns if c.startswith("postoSUF_")]
        if cols_suf:
            print("\n  rótulo = 'chunk suficiente' por BERTScore (item 20, não degenerado):")
            print(dpost.groupby(GRP)[cols_suf].mean().to_string(float_format=_fmt))

        # teste pareado entre entropia e cada outro escore
        try:
            from scipy.stats import wilcoxon
            print("\n  Wilcoxon pareado (posto da entropia x posto do outro escore):")
            for (modelo, crit), sub in dpost.groupby(GRP):
                for col, _, rotulo in ESCORES[1:]:
                    a, b = "posto_entropia_media", f"posto_{col}"
                    if a in sub and b in sub:
                        par = sub[[a, b]].dropna()
                        if len(par) >= 6 and (par[a] != par[b]).any():
                            st = wilcoxon(par[a], par[b])
                            print(f"    [{crit:<5}] {modelo:<28} vs {rotulo:<45} "
                                  f"n={len(par)}  p={st.pvalue:.4f}  "
                                  f"delta={float((par[a] - par[b]).mean()):+.2f}")

            # A x B: o posto do gold é a ÚNICA estatística de entropia que se
            # compara entre as modalidades. É invariante a reescala monotônica,
            # então não importa que as duas gerem distribuições diferentes.
            if len(criterios) > 1 and "posto_entropia_media" in dpost.columns:
                print("\n  Modalidade A x B -- posto do gold sob a entropia, pareado "
                      "por pergunta:")
                piv = dpost.pivot_table(index=["retriever", "modelo", "pergunta_idx"],
                                        columns="criterio", values="posto_entropia_media")
                if {"regex", "json"} <= set(piv.columns):
                    for modelo, sub in piv.groupby(level=[0, 1]):
                        par = sub[["regex", "json"]].dropna()
                        if not len(par):
                            continue
                        d_med = float((par["json"] - par["regex"]).mean())
                        p = (wilcoxon(par["regex"], par["json"]).pvalue
                             if len(par) >= 6 and (par["regex"] != par["json"]).any()
                             else float("nan"))
                        print(f"    {str(modelo):<34} n={len(par):<4} "
                              f"regex={par['regex'].mean():.2f}  "
                              f"json={par['json'].mean():.2f}  "
                              f"delta={d_med:+.2f}  p={p:.4f}")
                    print("    'n' conta só perguntas que sobreviveram aos filtros NAS")
                    print("    DUAS modalidades. Se n cair muito abaixo do total, o")
                    print("    pareamento está sobre um subconjunto -- reporte n.")
        except Exception as e:  # noqa: BLE001
            print(f"  [aviso] Wilcoxon indisponível ({type(e).__name__}: {e})")

        # estratificado (item 22c)
        est = df.drop_duplicates(["retriever", "modelo", "pergunta_idx"])[
            ["retriever", "modelo", "pergunta_idx", "estrato"]]
        dpost = dpost.merge(est, on=["retriever", "modelo", "pergunta_idx"], how="left")
        print("\n  por estrato (ver item 22c):")
        print(dpost.groupby(GRP + ["estrato"])[cols_posto].mean()
              .to_string(float_format=_fmt))

        dpost.to_parquet(os.path.join(comum.DIR_SAIDAS, "postos_bloco2.parquet"), index=False)
        tabelas_extra["postos_bloco2"] = dpost

        # ---- item 21b: nula ESTRATIFICADA por validade ---------------
        # A nula do item 18 permuta as 10 posições livremente, supondo que a
        # entropia é permutável entre elas. O AUC da seção anterior mostra que
        # não é. Aqui a permutação acontece DENTRO de cada estrato de validade:
        # preserva o acoplamento entropia-validade e pergunta o que de fato
        # interessa -- o escore sabe algo sobre o gold ALÉM da validade?
        print("\n--- item 21b [PRINCIPAL]: nula ingênua x nula ESTRATIFICADA por validade ---")
        print(f"  permutação, {comum.N_PERM_NULA} reamostragens, semente fixa. "
              f"p unilateral.")
        linhas_nula = []
        for (grp_ret, modelo, crit), sub in ent_ok.groupby(GRP):
            grupos, V, gold_val, obs_amr = [], [], [], []
            for _, q in sub.groupby("pergunta_idx"):
                q = q.dropna(subset=["entropia_media"])
                if len(q) < 2 or not q["is_gold"].any():
                    continue
                grupos.append((q["entropia_media"].to_numpy(float),
                               q["is_gold"].to_numpy(bool),
                               q["valida"].to_numpy(bool)))
                V.append(int(q["valida"].sum()))
                gold_val.append(bool(q.loc[q["is_gold"], "valida"].any()))
                qv = q[q["valida"]]
                if len(qv):
                    obs_amr.append(bool(qv.loc[qv["entropia_media"].idxmin(), "is_gold"]))
            if len(grupos) < 3:
                continue
            postos_obs = []
            argmin_obs = []
            for esc, g_, _v in grupos:
                ordem = np.argsort(esc, kind="stable")
                pp = np.empty(len(esc), int); pp[ordem] = np.arange(1, len(esc) + 1)
                postos_obs.append(int(pp[g_].min()))
                argmin_obs.append(bool(g_[np.argmin(esc)]))
            o_posto, o_argmin = float(np.mean(postos_obs)), float(np.mean(argmin_obs))

            linha = {"retriever": grp_ret, "modelo": modelo, "criterio": crit,
                     "n_perguntas": len(grupos),
                     "posto_obs": o_posto, "argmin_obs_%": 100 * o_argmin}
            for estrat, suf in ((False, "ingenua"), (True, "estratif")):
                pt, am = comum.amostrar_nula_estratificada(
                    grupos, sinal=-1, estratificar=estrat)
                linha[f"posto_nula_{suf}"] = float(pt.mean(axis=1).mean())
                linha[f"p_posto_{suf}"] = comum.p_valor_nula(
                    o_posto, pt.mean(axis=1), menor_e_melhor=True)
                linha[f"p_argmin_{suf}"] = comum.p_valor_nula(
                    o_argmin, am.mean(axis=1), menor_e_melhor=False)
            # secundária pré-declarada: restrita às posições válidas
            linha["restrito_obs_%"] = 100 * float(np.mean(obs_amr)) if obs_amr else np.nan
            linha["restrito_nula_%"] = 100 * comum.nula_restrita_exata(V, gold_val)
            linha["V_igual_1"] = int(sum(1 for x in V if x == 1))
            linhas_nula.append(linha)

        if linhas_nula:
            dn = pd.DataFrame(linhas_nula)
            print(dn[["retriever", "modelo", "criterio", "n_perguntas", "posto_obs",
                      "posto_nula_ingenua", "p_posto_ingenua",
                      "posto_nula_estratif", "p_posto_estratif"]]
                  .to_string(index=False, float_format=_fmt))
            print("\n  a mesma coisa para 'menor entropia é o gold', e a secundária restrita:")
            print(dn[["retriever", "modelo", "criterio", "argmin_obs_%", "p_argmin_ingenua",
                      "p_argmin_estratif", "restrito_obs_%", "restrito_nula_%",
                      "V_igual_1"]].to_string(index=False, float_format=_fmt))
            print("\n  COMO LER. `p_posto_estratif` é a leitura PRINCIPAL: ela já")
            print("  desconta tudo o que a estrutura de validade explica sozinha. Se a")
            print("  ingênua e a estratificada discordam, a ingênua é a errada -- ela")
            print("  supõe permutabilidade que o AUC da seção anterior nega.")
            print("  `restrito_nula_%` é média(1/V), NÃO 1/média(V): com V=1 a")
            print("  estatística restrita vale 1 por construção, e a coluna V_igual_1")
            print("  diz em quantas perguntas isso acontece. Se V_igual_1 for grande, a")
            print("  leitura restrita é dominada por perguntas degeneradas.")
            tabelas_extra["nula_estratificada_bloco2"] = dn
            p_estrat = dn.set_index(["retriever", "modelo", "criterio"])[
                ["p_argmin_estratif", "p_posto_estratif"]]

        # ---- item 19: dose-resposta analítica -----------------------
        print("\n--- item 19: dose-resposta da posição do gold (analítica, sem GPU) ---")
        print("  Depende de `valida`, que é a decisão do critério de parada, então a")
        print("  curva é POR MODALIDADE. Usa `ent` inteiro (e não ent_ok): a decisão")
        print("  de parada existe mesmo onde a entropia foi descartada.")
        curvas = []
        for (grp_ret, modelo, crit, pidx), sub in ent.groupby(GRP + ["pergunta_idx"]):
            if int(sub["is_gold"].sum()) != 1:
                continue                    # regra declarada: só single-gold
            g_val = bool(sub.loc[sub["is_gold"], "valida"].iloc[0])
            m = int(sub.loc[~sub["is_gold"], "valida"].sum())
            r = _dose_resposta(g_val, m)
            if r is None:
                continue
            p_stop, e_d = r
            for j in range(comum.K_TOP):
                curvas.append({"retriever": grp_ret, "modelo": modelo,
                               "criterio": crit, "pergunta_idx": pidx,
                               "posicao_gold": j + 1,
                               "P_parar_no_gold": p_stop[j], "E_D": e_d[j]})
        if curvas:
            dc = pd.DataFrame(curvas)
            print(f"  (n = {dc['pergunta_idx'].nunique()} perguntas com exatamente um "
                  f"gold entre os 10)")
            print(dc.groupby(GRP + ["posicao_gold"])[["P_parar_no_gold", "E_D"]]
                  .mean().unstack([0, 1]).to_string(float_format=_fmt))
            dc.to_parquet(os.path.join(comum.DIR_SAIDAS, "dose_resposta_bloco2.parquet"),
                          index=False)
            tabelas_extra["dose_resposta_bloco2"] = dc

        ent.to_parquet(os.path.join(comum.DIR_SAIDAS,
                                    "consolidado_bloco2_entropia.parquet"), index=False)

    # ---- tabela final do Bloco 2 -------------------------------------
    print("\n--- tabela do Bloco 2, por modalidade de parada ---")
    linhas = []
    for (grp_ret, modelo, crit), sub in df.groupby(GRP):
        n_inf = sub["n_inferencias"].sum() if "n_inferencias" in sub else 0
        linhas.append({
            "retriever": grp_ret, "modelo": modelo, "criterio": crit, "n": len(sub),
            "acaso_%": 100 * sub["n_golds_no_top10"].mean() / comum.K_TOP,
            "min_entropia_e_gold_% [SECUNDÁRIA]": 100 * sub["min_entropia_e_gold"].mean(),
            "P@1_retriever_%": 100 * sub["max_similaridade_e_gold"].mean(),
            "concord_D_entropia_%": 100 * sub["concordancia_D_entropia"].mean(),
            "acertou_chunk_gold_%": 100 * sub["acertou_chunk_gold"].mean(),
            "n_pos_validas_media": sub["n_posicoes_validas"].mean(),
            "D_embaralhado_mediana": sub["D_embaralhado"].median(),
            "conformidade_%": (100 * sub["n_pos_conformes"].sum() / n_inf
                               if crit == "json" and n_inf else np.nan),
            "janela_alinhada_%": (100 * sub["n_pos_janela_json_alinhada"].sum() / n_inf
                                  if crit == "json" and n_inf
                                  and "n_pos_janela_json_alinhada" in sub else np.nan),
            "S_f1_medio": sub["S_bertscore_f1"].mean() if "S_bertscore_f1" in sub else np.nan,
        })
    tabela = pd.DataFrame(linhas)
    # `min_entropia_e_gold` NÃO viaja sozinha. Ela é degenerada sempre que a
    # entropia é circular (sob separação total vale 0 por construção), e um
    # número desses copiado para a tese sem o p ao lado vira afirmação falsa
    # nos dois sentidos. O p estratificado do item 21b vai grudado na coluna.
    if p_estrat is not None:
        tabela = tabela.merge(
            p_estrat.rename(columns={"p_argmin_estratif": "p_estratif [item 21b]"})
                    [["p_estratif [item 21b]"]].reset_index(),
            on=["retriever", "modelo", "criterio"], how="left")
        cols = list(tabela.columns)
        i = cols.index("min_entropia_e_gold_% [SECUNDÁRIA]")
        cols.insert(i + 1, cols.pop(cols.index("p_estratif [item 21b]")))
        tabela = tabela[cols]
    print(tabela.to_string(index=False, float_format=_fmt))
    print("  P@1_retriever_% é caracterização do retriever, não comparação entre")
    print("  modelos nem entre modalidades (item 22a) -- tem que ser igual em")
    print("  todas as linhas do mesmo corpus. A comparação principal está no item 21.")
    print("  min_entropia_e_gold NUNCA deve ser citada sem o p ao lado: sob")
    print("  circularidade ela é 0 (ou 1) por construção, e o p é o que distingue")
    print("  'não há sinal' de 'a estatística não tinha grau de liberdade'.")

    # ---- Modalidade A x B no Bloco 2, pareado por pergunta -----------
    if len(criterios) > 1:
        print("\n--- Modalidade A (regex) x Modalidade B (JSON), pareado por pergunta ---")
        comp = []
        for (grp_ret, modelo), sub in df.groupby(["retriever", "modelo"]):
            piv_d = sub.pivot_table(index="pergunta_idx", columns="criterio",
                                    values="D_embaralhado")
            piv_v = sub.pivot_table(index="pergunta_idx", columns="criterio",
                                    values="n_posicoes_validas")
            piv_g = sub.pivot_table(index="pergunta_idx", columns="criterio",
                                    values="acertou_chunk_gold")
            if not {"regex", "json"} <= set(piv_d.columns):
                continue
            amb = piv_d.dropna()
            linha = {
                "retriever": grp_ret, "modelo": modelo,
                "n_pareado_D": len(amb),
                "so_regex_parou": int((piv_d["regex"].notna() & piv_d["json"].isna()).sum()),
                "so_json_parou": int((piv_d["json"].notna() & piv_d["regex"].isna()).sum()),
                "D_regex": amb["regex"].mean() if len(amb) else np.nan,
                "D_json": amb["json"].mean() if len(amb) else np.nan,
                "delta_D": (amb["json"] - amb["regex"]).mean() if len(amb) else np.nan,
                "D_igual_%": (100 * (amb["json"] == amb["regex"]).mean()
                              if len(amb) else np.nan),
                "V_regex": piv_v["regex"].mean(),
                "V_json": piv_v["json"].mean(),
                "acertou_gold_regex_%": 100 * piv_g["regex"].mean(),
                "acertou_gold_json_%": 100 * piv_g["json"].mean(),
            }
            comp.append(linha)
        if comp:
            print(pd.DataFrame(comp).to_string(index=False, float_format=_fmt))
            print("  'n_pareado_D' conta só perguntas em que AS DUAS modalidades")
            print("  pararam; as censuras assimétricas estão nas duas colunas 'so_*'")
            print("  e NÃO podem ser lidas como D = 10.")
            print("  V é o número de posições declaradas válidas: se V_json < V_regex,")
            print("  o contrato estruturado tornou o modelo mais conservador -- é o")
            print("  efeito da Modalidade B sobre a parada, não sobre a resposta.")
            tabelas_extra["comparacao_modalidades_bloco2"] = pd.DataFrame(comp)

        js = df[df["criterio"] == "json"]
        if len(js) and "D_embaralhado_regra_alternativa" in js.columns:
            div = (js["D_embaralhado"].fillna(-1)
                   != js["D_embaralhado_regra_alternativa"].fillna(-1))
            print(f"\n  Modalidade B -- D difere sob 'sentinela OU regex no valor' em "
                  f"{int(div.sum())}/{len(js)} ({100 * div.mean():.1f}%).")
            print("  D_embaralhado é a Eq. 22 pura; a coluna _regra_alternativa é a")
            print("  leitura robusta, que também trata recusa em prosa dentro do valor.")

    destino = os.path.join(comum.DIR_SAIDAS, "consolidado_bloco2.parquet")
    df.to_parquet(destino, index=False)
    df.to_csv(destino.replace(".parquet", ".csv"), index=False)
    tabela.to_csv(os.path.join(comum.DIR_SAIDAS, "tabela_bloco2.csv"), index=False)
    for nome, t in tabelas_extra.items():
        t.to_csv(os.path.join(comum.DIR_SAIDAS, f"{nome}.csv"), index=False)
    print(f"\n  -> {destino}")
    return df


def resumo_metadata():
    arqs = sorted(glob.glob(os.path.join(comum.DIR_SAIDAS, "metadata_*.json")))
    if not arqs:
        return
    print(f"\n{'=' * 78}\n### REGISTRO DE AMBIENTE (item 12)\n{'=' * 78}")
    linhas = []
    for a in arqs:
        with open(a, encoding="utf-8") as f:
            m = json.load(f)
        linhas.append({
            "arquivo": os.path.basename(a),
            "modelo": m.get("tag_modelo"), "bloco": m.get("bloco"),
            "quant": m.get("quantizacao"), "vocab": m.get("vocab_size"),
            "transformers": (m.get("versoes") or {}).get("transformers"),
            "torch": (m.get("versoes") or {}).get("torch"),
            "div_argmax_%": (100 * m["taxa_divergencia_argmax"]
                             if m.get("taxa_divergencia_argmax") is not None else None),
            "inicio>0_%": (100 * m["fracao_posicoes_com_inicio_maior_que_zero"]
                           if m.get("fracao_posicoes_com_inicio_maior_que_zero") is not None
                           else None),
            # parâmetros que TÊM de ser iguais entre rodadas comparadas
            "retriever": m.get("retriever"),
            "max_new_tokens": m.get("max_new_tokens"),
            "d_max": m.get("d_max"),
            "coluna_texto": m.get("coluna_texto"),
        })
    tab = pd.DataFrame(linhas)
    print(tab.to_string(index=False, float_format=_fmt))

    # ---- rodadas comparáveis? ----------------------------------------
    # Com o experimento paralelizado em mais de uma máquina, o risco novo é
    # divergência de PARÂMETRO entre elas: uma roda com max_new_tokens=800 e
    # outra com 400, e a diferença que se atribui ao retriever é o teto de
    # tokens. A seção 9.32/9.33 mostrou que esse teto vale -0,077 a -0,138 de
    # BERTScore no braço fixo50 -- muito acima do efeito medido. O aviso de
    # truncamento do Bloco 1 NÃO pega isto: ele compara braços dentro de uma
    # rodada, não rodadas entre si.
    chaves = ["max_new_tokens", "d_max", "coluna_texto", "transformers", "torch"]
    div = {k: sorted({str(v) for v in tab[k].dropna().unique()})
           for k in chaves if k in tab.columns}
    div = {k: v for k, v in div.items() if len(v) > 1}
    if div:
        print("\n  <<< ATENÇÃO: as rodadas em saidas/ NÃO usaram os mesmos parâmetros:")
        for k, v in div.items():
            print(f"      {k}: {v}")
        print("      Qualquer comparação que cruze essas rodadas mistura o efeito")
        print("      estudado com a diferença de parâmetro. Isso é especialmente")
        print("      grave para `max_new_tokens` (ver README 9.32) e para")
        print("      `coluna_texto`. Refaça as rodadas divergentes ou compare só")
        print("      dentro de cada grupo homogêneo.")
    elif len(tab) > 1:
        print("\n  OK: todas as rodadas em saidas/ usaram os mesmos parâmetros de "
              "medição.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalcular", action="store_true",
                    help="refaz o BERTScore mesmo se a coluna já existir")
    ap.add_argument("--sem-bertscore-740", action="store_true",
                    help="pula o item 20 (é o passo lento desta análise)")
    ap.add_argument("--manter-janela-json-desalinhada", action="store_true",
                    help="não descarta as posições da Modalidade B em que a janela "
                         "de entropia ficou na sintaxe do JSON. Use para MEDIR o "
                         "efeito do filtro, não para reportar o número principal")
    args = ap.parse_args()

    consolidar_bloco1(args.recalcular)
    consolidar_bloco2(args.recalcular, com_bertscore_740=not args.sem_bertscore_740,
                      filtrar_janela_json=not args.manter_janela_json_desalinhada)
    resumo_metadata()
    print("\nPronto. Os arquivos consolidado_*.parquet/.csv são o que a análise usa.")


if __name__ == "__main__":
    main()
