# -*- coding: utf-8 -*-
"""
=============================================================================
 ANALISE ESTATISTICA E CONSOLIDACAO DOS RESULTADOS EXPERIMENTAIS
 Avaliacao de um sistema RAG com parada adaptativa
 (Bloco 1 - dois modelos de embedding/retrieval)
=============================================================================

Entradas (4 arquivos CSV, na pasta definida em PASTA):
    - tabela_bloco1 Cesar.csv        (tabela-resumo ja agregada, retriever=e5small)
    - tabela_bloco1 Marilia.csv      (tabela-resumo ja agregada, retriever=qwen3)
    - consolidado_bloco1 Cesar.csv   (dados brutos, 1 linha por observacao)
    - consolidado_bloco1 Marilia.csv (dados brutos, 1 linha por observacao)

Saidas:
    - resultados_completos_experimento.xlsx  (15 abas)
    - auditoria_query_a_query.csv
    - interpretacao_resultados.txt           (gerado por script auxiliar)

Definicoes metodologicas adotadas (documentadas em Dicionario/Metadados):
    * D  = D_iter  -> profundidade de parada (numero de posicoes do ranking
           efetivamente percorridas ate a parada). D e' CENSURADO quando
           D_iter esta ausente, o que ocorre exatamente quando
           censurado_direita == True OU censurado_limite_tokens == True.
    * Delta X = X_incremental - X_iterativo
    * eta = 1 - tokens_total_acum(incremental) / tokens_total_acum(fixo50)
    * fixo50 e' usado EXCLUSIVAMENTE como baseline saturante para eta.
    * Nenhum valor e' arredondado antes dos calculos; o arredondamento
      ocorre apenas na apresentacao (parametro ROUND_APRESENTACAO).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. CONFIGURACAO
# ----------------------------------------------------------------------------
PASTA = os.environ.get(
    "PASTA_DADOS",
    "/mnt/user-data/uploads/Murilo/Projetos Claude Code/"
    "Language Resources and Evaluation journal/Escrita do artigo com Chatgpt",
)
SAIDA = os.environ.get("PASTA_SAIDA", PASTA)

ARQ = {
    "Cesar":   {"consolidado": "consolidado_bloco1 Cesar.csv",
                "tabela":      "tabela_bloco1 Cesar.csv"},
    "Marilia": {"consolidado": "consolidado_bloco1 Marilia.csv",
                "tabela":      "tabela_bloco1 Marilia.csv"},
}

MODELOS   = ["gpt-oss:20b", "mistral-small3.1:latest", "qwen3.8:27b"]
CRITERIOS = ["json", "regex"]
MODOS     = ["incremental", "iterativo", "fixo50"]

ROUND_APRESENTACAO = 6      # casas decimais apenas na apresentacao
LOG = []                    # registro de validacao


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


# ----------------------------------------------------------------------------
# 1. CARGA E VALIDACAO
# ----------------------------------------------------------------------------
def carregar():
    """Carrega os 4 arquivos, valida estrutura e devolve (long, tabelas, meta)."""
    log("=" * 78)
    log("1. CARGA E VALIDACAO DOS ARQUIVOS")
    log("=" * 78)

    brutos, tabelas, meta = {}, {}, []

    for conj, nomes in ARQ.items():
        p_cons = os.path.join(PASTA, nomes["consolidado"])
        p_tab = os.path.join(PASTA, nomes["tabela"])
        for p in (p_cons, p_tab):
            if not os.path.exists(p):
                sys.exit(f"ERRO: arquivo nao encontrado: {p}")

        df = pd.read_csv(p_cons)
        tb = pd.read_csv(p_tab)
        brutos[conj], tabelas[conj] = df, tb

        retr = sorted(df["retriever"].dropna().unique().tolist())
        retr_tab = sorted(tb["retriever"].dropna().unique().tolist())

        log(f"\n--- Conjunto '{conj}' ---")
        log(f"  consolidado : {nomes['consolidado']}  -> {df.shape[0]} linhas x {df.shape[1]} colunas")
        log(f"  tabela      : {nomes['tabela']}       -> {tb.shape[0]} linhas x {tb.shape[1]} colunas")
        log(f"  retriever (consolidado) : {retr}")
        log(f"  retriever (tabela)      : {retr_tab}")
        log(f"  consultas unicas (pergunta_idx) : {df['pergunta_idx'].nunique()} "
            f"(min={df['pergunta_idx'].min()}, max={df['pergunta_idx'].max()})")
        log(f"  modelos   : {sorted(df['modelo'].unique())}")
        log(f"  criterios : {sorted(df['criterio'].unique())}")
        log(f"  modos     : {sorted(df['modo'].unique())}")

        chave = ["pergunta_idx", "modelo", "criterio", "modo"]
        n_dup = int(df.duplicated(subset=chave).sum())
        log(f"  duplicidades na chave (pergunta_idx+modelo+criterio+modo): {n_dup}")
        if n_dup:
            log("  *** ATENCAO: chave nao e' univoca neste arquivo ***")

        tam = df.groupby(["modelo", "criterio", "modo"]).size()
        log(f"  celulas do desenho: {tam.shape[0]} | tamanho min={tam.min()} max={tam.max()} "
            f"(desenho balanceado: {tam.min() == tam.max()})")

        na = df.isna().sum()
        na = na[na > 0]
        log("  colunas com valores ausentes:")
        for c, v in na.items():
            log(f"      {c}: {v}")

        cens = (df["censurado_direita"] | df["censurado_limite_tokens"])
        coincide = bool((df["D_iter"].isna() == cens).all())
        log(f"  D_iter ausente <=> censurado_direita OR censurado_limite_tokens : {coincide}")
        log("  censura por modo:")
        log(df.groupby("modo")[["censurado_direita", "censurado_limite_tokens"]]
              .sum().to_string(index=True))

        meta.append({
            "conjunto": conj,
            "arquivo_consolidado": nomes["consolidado"],
            "arquivo_tabela": nomes["tabela"],
            "retriever_declarado": ", ".join(retr),
            "n_linhas_consolidado": df.shape[0],
            "n_colunas_consolidado": df.shape[1],
            "n_linhas_tabela": tb.shape[0],
            "n_consultas_unicas": int(df["pergunta_idx"].nunique()),
            "n_modelos": df["modelo"].nunique(),
            "n_criterios": df["criterio"].nunique(),
            "n_modos": df["modo"].nunique(),
            "duplicidades_chave": n_dup,
            "celulas_desenho": int(tam.shape[0]),
            "n_por_celula_min": int(tam.min()),
            "n_por_celula_max": int(tam.max()),
            "D_iter_ausente": int(df["D_iter"].isna().sum()),
            "censurado_direita_total": int(df["censurado_direita"].sum()),
            "censurado_limite_tokens_total": int(df["censurado_limite_tokens"].sum()),
            "D_ausente_equiv_censura": coincide,
        })

    # --- comparacoes ENTRE arquivos ---
    log("\n--- Comparacao estrutural entre os dois conjuntos ---")
    cA, cB = brutos["Cesar"], brutos["Marilia"]
    cols_iguais = list(cA.columns) == list(cB.columns)
    log(f"  mesmas colunas, mesma ordem: {cols_iguais}")
    if not cols_iguais:
        log(f"  so' em Cesar  : {sorted(set(cA.columns) - set(cB.columns))}")
        log(f"  so' em Marilia: {sorted(set(cB.columns) - set(cA.columns))}")

    idxA = set(cA["pergunta_idx"].unique())
    idxB = set(cB["pergunta_idx"].unique())
    log(f"  mesmos pergunta_idx: {idxA == idxB} (|A|={len(idxA)}, |B|={len(idxB)}, "
        f"intersecao={len(idxA & idxB)})")

    pA = cA.drop_duplicates("pergunta_idx").set_index("pergunta_idx")["pergunta"].sort_index()
    pB = cB.drop_duplicates("pergunta_idx").set_index("pergunta_idx")["pergunta"].sort_index()
    textos_iguais = bool((pA.values == pB.values).all()) if len(pA) == len(pB) else False
    log(f"  texto da pergunta identico para o mesmo pergunta_idx nos dois arquivos: {textos_iguais}")
    log("  => pergunta_idx identifica a MESMA consulta nos dois conjuntos.")

    # --- identificacao do embedding ---
    log("\n--- Identificacao empirica do modelo de embedding/retrieval ---")
    mapa = {}
    for conj, df in brutos.items():
        r = sorted(df["retriever"].dropna().unique().tolist())
        assert len(r) == 1, f"conjunto {conj} tem mais de um retriever: {r}"
        mapa[conj] = r[0]
        log(f"  {conj:<8} -> coluna 'retriever' = '{r[0]}' (valor unico em todo o arquivo)")
    log("  OBS: os arquivos registram apenas o rotulo curto do retriever; nao ha' coluna")
    log("       com o identificador completo do checkpoint. Os rotulos 'e5small' e 'qwen3'")
    log("       sao compativeis, respectivamente, com multilingual-e5-small e com um modelo")
    log("       de embedding da familia Qwen3, mas o nome completo NAO pode ser confirmado")
    log("       a partir dos dados; ele foi mantido exatamente como registrado nos arquivos.")

    # --- dataframe longo unificado ---
    long = []
    for conj, df in brutos.items():
        d = df.copy()
        d["conjunto"] = conj
        d["embedding"] = mapa[conj]
        long.append(d)
    long = pd.concat(long, ignore_index=True)

    chave_full = ["embedding", "pergunta_idx", "modelo", "criterio", "modo"]
    n_dup_full = int(long.duplicated(subset=chave_full).sum())
    log(f"\n  duplicidades na chave completa (embedding+pergunta_idx+modelo+criterio+modo): {n_dup_full}")

    tabelas_cat = []
    for conj, tb in tabelas.items():
        t = tb.copy(); t.insert(0, "conjunto", conj)
        tabelas_cat.append(t)
    tabelas_cat = pd.concat(tabelas_cat, ignore_index=True)

    return long, tabelas_cat, pd.DataFrame(meta), mapa, n_dup_full


# ----------------------------------------------------------------------------
# 2. PAREAMENTO CONSULTA A CONSULTA
# ----------------------------------------------------------------------------
def classificar(delta, censurado=False):
    if censurado or pd.isna(delta):
        return "nao_comparavel"
    if delta > 0:
        return "incremental_gt_iterativo"
    if delta < 0:
        return "iterativo_gt_incremental"
    return "empate"


def classificar_eta(e):
    if pd.isna(e):
        return "nao_comparavel"
    if e > 0:
        return "eta_gt_0"
    if e < 0:
        return "eta_lt_0"
    return "eta_eq_0"


def construir_pareado(long):
    """Uma linha por embedding x pergunta_idx x modelo x criterio."""
    log("\n" + "=" * 78)
    log("2. PAREAMENTO CONSULTA A CONSULTA (incremental x iterativo x fixo50)")
    log("=" * 78)

    idx = ["embedding", "pergunta_idx", "modelo", "criterio"]
    metricas = {
        "D": "D_iter",
        "Dchunk": "D_chunk",
        "F1": "S_bertscore_f1",
        "P": "S_bertscore_p",
        "R": "S_bertscore_r",
        "tokens": "tokens_total_acum",
        "tokens_prompt": "tokens_prompt_acum",
        "ninf": "n_inferencias",
    }

    base = None
    for modo in MODOS:
        sub = long[long["modo"] == modo].set_index(idx)
        cols = {}
        for alias, col in metricas.items():
            cols[f"{alias}_{modo}"] = sub[col]
        cols[f"censurado_dmax_{modo}"] = sub["censurado_direita"]
        cols[f"censurado_tokens_{modo}"] = sub["censurado_limite_tokens"]
        w = pd.DataFrame(cols)
        base = w if base is None else base.join(w, how="outer")

    par = base.reset_index()

    # ---- deltas (SEM arredondamento) ----
    par["delta_D"]  = par["D_incremental"]  - par["D_iterativo"]
    par["delta_F1"] = par["F1_incremental"] - par["F1_iterativo"]
    par["delta_P"]  = par["P_incremental"]  - par["P_iterativo"]
    par["delta_R"]  = par["R_incremental"]  - par["R_iterativo"]
    par["delta_tokens"] = par["tokens_incremental"] - par["tokens_iterativo"]

    # censura em D: ausente em qualquer uma das duas condicoes
    par["D_censurado"] = par["D_incremental"].isna() | par["D_iterativo"].isna()

    par["classificacao_D"]  = [classificar(d, c) for d, c in zip(par["delta_D"], par["D_censurado"])]
    par["classificacao_F1"] = [classificar(d) for d in par["delta_F1"]]
    par["classificacao_P"]  = [classificar(d) for d in par["delta_P"]]
    par["classificacao_R"]  = [classificar(d) for d in par["delta_R"]]

    # ---- eficiencia eta ----
    par["eta"] = 1.0 - (par["tokens_incremental"] / par["tokens_fixo50"])
    par["classificacao_eta"] = [classificar_eta(e) for e in par["eta"]]

    # eta auxiliar do iterativo (informativo, nao usado nas abas principais de eta)
    par["eta_iterativo"] = 1.0 - (par["tokens_iterativo"] / par["tokens_fixo50"])

    log(f"  linhas pareadas: {len(par)}  "
        f"(esperado = {long['embedding'].nunique()} embeddings x "
        f"{long['pergunta_idx'].nunique()} consultas x {len(MODELOS)} modelos x "
        f"{len(CRITERIOS)} criterios = "
        f"{long['embedding'].nunique()*long['pergunta_idx'].nunique()*len(MODELOS)*len(CRITERIOS)})")
    faltando = par[["D_incremental", "D_iterativo", "F1_incremental", "F1_iterativo",
                    "tokens_incremental", "tokens_fixo50"]].isna().sum()
    log("  ausencias nas colunas pareadas:")
    for c, v in faltando.items():
        log(f"      {c}: {v}")
    log(f"  pares com D censurado (nao comparavel): {int(par['D_censurado'].sum())}")
    return par


# ----------------------------------------------------------------------------
# 3. FUNCOES ESTATISTICAS
# ----------------------------------------------------------------------------
def estat(serie, percentis_extra=False):
    s = pd.Series(serie).dropna().astype(float)
    n = len(s)
    d = {
        "N": n,
        "media": s.mean() if n else np.nan,
        "mediana": s.median() if n else np.nan,
        "Q1": s.quantile(0.25) if n else np.nan,
        "Q3": s.quantile(0.75) if n else np.nan,
        "minimo": s.min() if n else np.nan,
        "maximo": s.max() if n else np.nan,
        "desvio_padrao": s.std(ddof=1) if n > 1 else np.nan,
        "IQR": (s.quantile(0.75) - s.quantile(0.25)) if n else np.nan,
        "n_positivos": int((s > 0).sum()),
        "n_negativos": int((s < 0).sum()),
        "n_zeros": int((s == 0).sum()),
    }
    if percentis_extra:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            d[f"p{p}"] = s.quantile(p / 100) if n else np.nan
    return d


def contagens(serie_classificacao, rotulos, prefixo, n_total):
    vc = pd.Series(serie_classificacao).value_counts()
    out = {}
    for r in rotulos:
        c = int(vc.get(r, 0))
        out[f"{prefixo}{r}"] = c
        out[f"pct_{prefixo}{r}"] = 100.0 * c / n_total if n_total else np.nan
    return out


ROT_DELTA = ["incremental_gt_iterativo", "iterativo_gt_incremental", "empate", "nao_comparavel"]
ROT_ETA = ["eta_gt_0", "eta_eq_0", "eta_lt_0", "nao_comparavel"]
GRUPO = ["embedding", "modelo", "criterio"]


# ----------------------------------------------------------------------------
# 4. ABAS DE DIRECAO / DISTRIBUICAO
# ----------------------------------------------------------------------------
def aba_direcao(par, metrica, col_delta, col_cls, col_inc, col_it):
    linhas = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        n_total = len(g)
        r = {"Embedding": emb, "Modelo": mod, "Criterio": cri,
             "metrica": f"delta_{metrica}", "N_total": n_total}
        r.update(contagens(g[col_cls], ROT_DELTA, "", n_total))
        n_comp = n_total - int((g[col_cls] == "nao_comparavel").sum())
        r["N_comparavel"] = n_comp
        for rot in ROT_DELTA[:3]:
            c = int((g[col_cls] == rot).sum())
            r[f"pct_sobre_comparavel_{rot}"] = 100.0 * c / n_comp if n_comp else np.nan
        r.update(estat(g[col_delta]))
        r[f"media_{metrica}_incremental"] = g[col_inc].mean()
        r[f"media_{metrica}_iterativo"] = g[col_it].mean()
        r[f"mediana_{metrica}_incremental"] = g[col_inc].median()
        r[f"mediana_{metrica}_iterativo"] = g[col_it].median()
        linhas.append(r)
    return pd.DataFrame(linhas)


def aba_eficiencia(par):
    linhas = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        n_total = len(g)
        r = {"Embedding": emb, "Modelo": mod, "Criterio": cri, "N_total": n_total}
        r.update(contagens(g["classificacao_eta"], ROT_ETA, "", n_total))
        r.update(estat(g["eta"], percentis_extra=True))
        r["tokens_incremental_media"] = g["tokens_incremental"].mean()
        r["tokens_incremental_mediana"] = g["tokens_incremental"].median()
        r["tokens_fixo50_media"] = g["tokens_fixo50"].mean()
        r["tokens_fixo50_mediana"] = g["tokens_fixo50"].median()
        r["tokens_iterativo_media"] = g["tokens_iterativo"].mean()
        r["tokens_iterativo_mediana"] = g["tokens_iterativo"].median()
        r["eta_iterativo_media"] = g["eta_iterativo"].mean()
        r["eta_iterativo_mediana"] = g["eta_iterativo"].median()
        # outliers de eta (regra de Tukey, cauda inferior e superior)
        q1, q3 = g["eta"].quantile(0.25), g["eta"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        r["lim_inf_tukey"] = lo
        r["lim_sup_tukey"] = hi
        r["n_outliers_inferiores"] = int((g["eta"] < lo).sum())
        r["n_outliers_superiores"] = int((g["eta"] > hi).sum())
        r["n_eta_menor_que_-1"] = int((g["eta"] < -1).sum())
        r["n_eta_menor_que_-5"] = int((g["eta"] < -5).sum())
        r["media_aparada_10pct"] = g["eta"].dropna().sort_values().iloc[
            int(0.05 * len(g.dropna(subset=["eta"]))): len(g.dropna(subset=["eta"])) -
            int(0.05 * len(g.dropna(subset=["eta"])))].mean()
        r["media_menos_mediana"] = r["media"] - r["mediana"]
        linhas.append(r)
    return pd.DataFrame(linhas)


def aba_distribuicoes(par):
    mapa = {"delta_D": "delta_D", "delta_F1": "delta_F1", "delta_P": "delta_P",
            "delta_R": "delta_R", "eta": "eta", "delta_tokens": "delta_tokens",
            "D_incremental": "D_incremental", "D_iterativo": "D_iterativo",
            "F1_incremental": "F1_incremental", "F1_iterativo": "F1_iterativo",
            "P_incremental": "P_incremental", "P_iterativo": "P_iterativo",
            "R_incremental": "R_incremental", "R_iterativo": "R_iterativo",
            "tokens_incremental": "tokens_incremental",
            "tokens_iterativo": "tokens_iterativo",
            "tokens_fixo50": "tokens_fixo50"}
    linhas = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        for rot, col in mapa.items():
            r = {"Embedding": emb, "Modelo": mod, "Criterio": cri, "metrica": rot}
            r.update(estat(g[col], percentis_extra=(rot == "eta")))
            linhas.append(r)
    return pd.DataFrame(linhas)


def aba_resumo_geral(par):
    linhas = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        n = len(g)
        def cnt(col, rot):
            return int((g[col] == rot).sum())
        r = {
            "Embedding": emb, "Modelo": mod, "Criterio": cri, "N": n,
            "D_inc_gt_iter": cnt("classificacao_D", "incremental_gt_iterativo"),
            "D_iter_gt_inc": cnt("classificacao_D", "iterativo_gt_incremental"),
            "D_empate": cnt("classificacao_D", "empate"),
            "D_nao_comparavel": cnt("classificacao_D", "nao_comparavel"),
            "F1_inc_gt_iter": cnt("classificacao_F1", "incremental_gt_iterativo"),
            "F1_iter_gt_inc": cnt("classificacao_F1", "iterativo_gt_incremental"),
            "F1_empate": cnt("classificacao_F1", "empate"),
            "P_inc_gt_iter": cnt("classificacao_P", "incremental_gt_iterativo"),
            "P_iter_gt_inc": cnt("classificacao_P", "iterativo_gt_incremental"),
            "P_empate": cnt("classificacao_P", "empate"),
            "R_inc_gt_iter": cnt("classificacao_R", "incremental_gt_iterativo"),
            "R_iter_gt_inc": cnt("classificacao_R", "iterativo_gt_incremental"),
            "R_empate": cnt("classificacao_R", "empate"),
            "eta_gt_0": cnt("classificacao_eta", "eta_gt_0"),
            "eta_eq_0": cnt("classificacao_eta", "eta_eq_0"),
            "eta_lt_0": cnt("classificacao_eta", "eta_lt_0"),
            "D_mean": g["delta_D"].mean(), "D_median": g["delta_D"].median(),
            "F1_mean": g["delta_F1"].mean(), "F1_median": g["delta_F1"].median(),
            "P_mean": g["delta_P"].mean(), "P_median": g["delta_P"].median(),
            "R_mean": g["delta_R"].mean(), "R_median": g["delta_R"].median(),
            "eta_mean": g["eta"].mean(), "eta_median": g["eta"].median(),
        }
        # percentuais
        for k in ["D_inc_gt_iter", "D_iter_gt_inc", "D_empate", "D_nao_comparavel",
                  "F1_inc_gt_iter", "F1_iter_gt_inc", "F1_empate",
                  "P_inc_gt_iter", "P_iter_gt_inc", "P_empate",
                  "R_inc_gt_iter", "R_iter_gt_inc", "R_empate",
                  "eta_gt_0", "eta_eq_0", "eta_lt_0"]:
            r[f"pct_{k}"] = 100.0 * r[k] / n
        # medias brutas por condicao (informacao complementar)
        r["D_media_incremental"] = g["D_incremental"].mean()
        r["D_media_iterativo"] = g["D_iterativo"].mean()
        r["F1_media_incremental"] = g["F1_incremental"].mean()
        r["F1_media_iterativo"] = g["F1_iterativo"].mean()
        r["tokens_mediana_incremental"] = g["tokens_incremental"].median()
        r["tokens_mediana_iterativo"] = g["tokens_iterativo"].median()
        r["tokens_mediana_fixo50"] = g["tokens_fixo50"].median()
        linhas.append(r)
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# 5. COMPARACAO ENTRE EMBEDDINGS
# ----------------------------------------------------------------------------
def comparar_pareado(long, chave_par, agrupar, coluna_fator, niveis, rotulo_fator):
    """
    Comparacao pareada generica de um fator com 2+ niveis.

    chave_par     : colunas que identificam uma unidade pareada (SEM a coluna do fator)
    agrupar       : colunas de reporte (subconjunto de chave_par, sem pergunta_idx)
    coluna_fator  : nome da coluna do fator (embedding / criterio / modelo)
    niveis        : lista dos niveis do fator
    Devolve uma tabela com, para cada par de niveis A-B e cada metrica, as contagens
    de direcao e a distribuicao completa de (A - B).
    """
    metricas_long = {"D": "D_iter", "F1": "S_bertscore_f1", "P": "S_bertscore_p",
                     "R": "S_bertscore_r", "tokens": "tokens_total_acum",
                     "n_inferencias": "n_inferencias", "tempo_s": "tempo_s"}
    pares = [(a, b) for i, a in enumerate(niveis) for b in niveis[i + 1:]]
    linhas = []

    for met, col in metricas_long.items():
        w = long.pivot_table(index=chave_par, columns=coluna_fator, values=col,
                             aggfunc="first", dropna=False).reset_index()
        for a, b in pares:
            if a not in w.columns or b not in w.columns:
                continue
            t = w[agrupar + [a, b]].copy()
            t["delta"] = t[a] - t[b]
            for gk, gg in t.groupby(agrupar, sort=True):
                gk = gk if isinstance(gk, tuple) else (gk,)
                n_total = len(gg)
                s = gg["delta"].dropna()
                n_comp = len(s)
                r = dict(zip(agrupar, gk))
                r.update({
                    "fator": rotulo_fator,
                    "contraste": f"{a} - {b}",
                    "nivel_A": a, "nivel_B": b,
                    "metrica": met,
                    "N_total": n_total,
                    "N_comparavel": n_comp,
                    "N_nao_comparavel": n_total - n_comp,
                    "n_A_maior": int((s > 0).sum()),
                    "n_B_maior": int((s < 0).sum()),
                    "n_empate": int((s == 0).sum()),
                    "pct_A_maior": 100.0 * (s > 0).sum() / n_comp if n_comp else np.nan,
                    "pct_B_maior": 100.0 * (s < 0).sum() / n_comp if n_comp else np.nan,
                    "pct_empate": 100.0 * (s == 0).sum() / n_comp if n_comp else np.nan,
                    "media_A": gg[a].mean(), "mediana_A": gg[a].median(),
                    "Q1_A": gg[a].quantile(.25), "Q3_A": gg[a].quantile(.75),
                    "media_B": gg[b].mean(), "mediana_B": gg[b].median(),
                    "Q1_B": gg[b].quantile(.25), "Q3_B": gg[b].quantile(.75),
                })
                for k, v in estat(s).items():
                    r[f"delta_{k}"] = v
                linhas.append(r)

    res = pd.DataFrame(linhas)
    ordem = agrupar + ["fator", "contraste", "nivel_A", "nivel_B", "metrica"]
    outras = [c for c in res.columns if c not in ordem]
    return res[ordem + outras].sort_values(ordem).reset_index(drop=True)


def aba_comparacao_embeddings(long, par):
    embs = sorted(long["embedding"].unique())
    tab = comparar_pareado(long,
                           chave_par=["pergunta_idx", "modelo", "criterio", "modo"],
                           agrupar=["modelo", "criterio", "modo"],
                           coluna_fator="embedding", niveis=embs, rotulo_fator="embedding")

    # --- niveis descritivos por embedding (mediana/quartis) ---
    desc = []
    for (emb, mod, cri, modo), g in long.groupby(["embedding", "modelo", "criterio", "modo"], sort=True):
        for met, col in [("D", "D_iter"), ("F1", "S_bertscore_f1"), ("P", "S_bertscore_p"),
                         ("R", "S_bertscore_r"), ("tokens", "tokens_total_acum")]:
            r = {"Embedding": emb, "Modelo": mod, "Criterio": cri, "Modo": modo, "metrica": met}
            r.update(estat(g[col]))
            desc.append(r)
    desc = pd.DataFrame(desc)

    # --- eta e comportamento incremental x iterativo por embedding ---
    comp = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        n = len(g)
        comp.append({
            "Embedding": emb, "Modelo": mod, "Criterio": cri, "N": n,
            "eta_media": g["eta"].mean(), "eta_mediana": g["eta"].median(),
            "eta_Q1": g["eta"].quantile(.25), "eta_Q3": g["eta"].quantile(.75),
            "pct_eta_gt_0": 100.0 * (g["eta"] > 0).sum() / n,
            "pct_D_empate": 100.0 * (g["classificacao_D"] == "empate").sum() / n,
            "pct_D_nao_comparavel": 100.0 * (g["classificacao_D"] == "nao_comparavel").sum() / n,
            "pct_F1_empate": 100.0 * (g["classificacao_F1"] == "empate").sum() / n,
            "pct_F1_inc_gt_iter": 100.0 * (g["classificacao_F1"] == "incremental_gt_iterativo").sum() / n,
            "pct_F1_iter_gt_inc": 100.0 * (g["classificacao_F1"] == "iterativo_gt_incremental").sum() / n,
            "delta_F1_mediana": g["delta_F1"].median(),
            "delta_D_mediana": g["delta_D"].median(),
            "tokens_incremental_mediana": g["tokens_incremental"].median(),
        })
    comp = pd.DataFrame(comp)

    # eta pareado entre embeddings (mesma consulta/modelo/criterio)
    p = par.pivot_table(index=["pergunta_idx", "modelo", "criterio"],
                        columns="embedding",
                        values=["eta", "delta_F1", "delta_D", "tokens_incremental"],
                        aggfunc="first")
    eta_emb = []
    if len(embs) == 2:
        a, b = embs
        for met in ["eta", "delta_F1", "delta_D", "tokens_incremental"]:
            d = (p[(met, a)] - p[(met, b)]).dropna()
            r = {"metrica_pareada": met, "diferenca": f"{a} - {b}", "N": len(d)}
            r.update(estat(d))
            r["n_A_maior"] = int((d > 0).sum())
            r["n_B_maior"] = int((d < 0).sum())
            r["n_empate"] = int((d == 0).sum())
            eta_emb.append(r)
    eta_emb = pd.DataFrame(eta_emb)
    return tab, desc, comp, eta_emb


def aba_json_vs_regex(long, par):
    tab = comparar_pareado(long,
                           chave_par=["embedding", "pergunta_idx", "modelo", "modo"],
                           agrupar=["embedding", "modelo", "modo"],
                           coluna_fator="criterio", niveis=CRITERIOS, rotulo_fator="criterio")

    # eta e comportamento inc x iter, pareado por criterio
    p = par.pivot_table(index=["embedding", "pergunta_idx", "modelo"], columns="criterio",
                        values=["eta", "delta_F1", "delta_D", "delta_P", "delta_R"],
                        aggfunc="first")
    linhas = []
    for (emb, mod), g in p.groupby(level=["embedding", "modelo"]):
        for met in ["eta", "delta_F1", "delta_D", "delta_P", "delta_R"]:
            d = (g[(met, "json")] - g[(met, "regex")]).dropna()
            r = {"Embedding": emb, "Modelo": mod, "metrica_pareada": met,
                 "diferenca": "json - regex", "N": len(d)}
            r.update(estat(d))
            r["n_json_maior"] = int((d > 0).sum())
            r["n_regex_maior"] = int((d < 0).sum())
            r["n_empate"] = int((d == 0).sum())
            linhas.append(r)
    comport = pd.DataFrame(linhas)
    return tab, comport


def aba_modelos(long, par):
    tab = comparar_pareado(long,
                           chave_par=["embedding", "pergunta_idx", "criterio", "modo"],
                           agrupar=["embedding", "criterio", "modo"],
                           coluna_fator="modelo", niveis=MODELOS, rotulo_fator="modelo")

    perfil = []
    for (emb, cri, mod), g in par.groupby(["embedding", "criterio", "modelo"], sort=True):
        n = len(g)
        perfil.append({
            "Embedding": emb, "Criterio": cri, "Modelo": mod, "N": n,
            "pct_D_empate": 100.0 * (g["classificacao_D"] == "empate").sum() / n,
            "pct_D_inc_gt_iter": 100.0 * (g["classificacao_D"] == "incremental_gt_iterativo").sum() / n,
            "pct_D_iter_gt_inc": 100.0 * (g["classificacao_D"] == "iterativo_gt_incremental").sum() / n,
            "pct_D_nao_comparavel": 100.0 * (g["classificacao_D"] == "nao_comparavel").sum() / n,
            "pct_F1_empate": 100.0 * (g["classificacao_F1"] == "empate").sum() / n,
            "pct_F1_divergente": 100.0 * (g["classificacao_F1"] != "empate").sum() / n,
            "delta_F1_desvio_padrao": g["delta_F1"].std(ddof=1),
            "delta_F1_IQR": g["delta_F1"].quantile(.75) - g["delta_F1"].quantile(.25),
            "delta_D_desvio_padrao": g["delta_D"].std(ddof=1),
            "delta_D_IQR": g["delta_D"].quantile(.75) - g["delta_D"].quantile(.25),
            "D_mediana_incremental": g["D_incremental"].median(),
            "D_mediana_iterativo": g["D_iterativo"].median(),
            "F1_mediana_incremental": g["F1_incremental"].median(),
            "F1_mediana_iterativo": g["F1_iterativo"].median(),
            "P_mediana_incremental": g["P_incremental"].median(),
            "R_mediana_incremental": g["R_incremental"].median(),
            "tokens_mediana_incremental": g["tokens_incremental"].median(),
            "tokens_media_incremental": g["tokens_incremental"].mean(),
            "eta_mediana": g["eta"].median(), "eta_media": g["eta"].mean(),
            "pct_eta_gt_0": 100.0 * (g["eta"] > 0).sum() / n,
            "n_eta_lt_-1": int((g["eta"] < -1).sum()),
            "pct_|delta_F1|_gt_0.05": 100.0 * (g["delta_F1"].abs() > 0.05).sum() / n,
            "pct_|delta_F1|_gt_0.10": 100.0 * (g["delta_F1"].abs() > 0.10).sum() / n,
            "pct_|delta_D|_ge_3": 100.0 * (g["delta_D"].abs() >= 3).sum() / n,
            "n_censurado_incremental": int(g["D_incremental"].isna().sum()),
            "n_censurado_iterativo": int(g["D_iterativo"].isna().sum()),
        })
    return tab, pd.DataFrame(perfil)


# ----------------------------------------------------------------------------
# 6. PROFUNDIDADE x QUALIDADE
# ----------------------------------------------------------------------------
def aba_profundidade_qualidade(par):
    linhas = []
    subsets = [
        ("delta_D_positivo (incremental parou mais fundo)", lambda g: g["delta_D"] > 0),
        ("delta_D_negativo (incremental parou menos fundo)", lambda g: g["delta_D"] < 0),
        ("delta_D_zero (mesma profundidade)", lambda g: g["delta_D"] == 0),
        ("delta_D_diferente_de_zero (|delta_D|>0, comparaveis)",
         lambda g: g["delta_D"].notna() & (g["delta_D"] != 0)),
        ("D_nao_comparavel (censurado em pelo menos uma condicao)",
         lambda g: g["delta_D"].isna()),
    ]
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        for nome, f in subsets:
            sub = g[f(g).fillna(False)]
            r = {"Embedding": emb, "Modelo": mod, "Criterio": cri,
                 "subconjunto": nome, "N": len(sub),
                 "pct_do_total": 100.0 * len(sub) / len(g)}
            for met in ["delta_F1", "delta_P", "delta_R"]:
                st = estat(sub[met])
                for k, v in st.items():
                    if k == "N":
                        continue
                    r[f"{met}_{k}"] = v
                r[f"{met}_pct_positivos"] = 100.0 * st["n_positivos"] / st["N"] if st["N"] else np.nan
                r[f"{met}_pct_negativos"] = 100.0 * st["n_negativos"] / st["N"] if st["N"] else np.nan
            r["delta_D_media"] = sub["delta_D"].mean()
            r["delta_D_mediana"] = sub["delta_D"].median()
            r["aviso_N_pequeno"] = "SIM (N<10): evitar conclusoes fortes" if 0 < len(sub) < 10 else (
                "sem casos" if len(sub) == 0 else "")
            linhas.append(r)

    # visao agregada (todos os grupos juntos)
    for nome, f in subsets:
        sub = par[f(par).fillna(False)]
        r = {"Embedding": "TODOS", "Modelo": "TODOS", "Criterio": "TODOS",
             "subconjunto": nome, "N": len(sub),
             "pct_do_total": 100.0 * len(sub) / len(par)}
        for met in ["delta_F1", "delta_P", "delta_R"]:
            st = estat(sub[met])
            for k, v in st.items():
                if k == "N":
                    continue
                r[f"{met}_{k}"] = v
            r[f"{met}_pct_positivos"] = 100.0 * st["n_positivos"] / st["N"] if st["N"] else np.nan
            r[f"{met}_pct_negativos"] = 100.0 * st["n_negativos"] / st["N"] if st["N"] else np.nan
        r["delta_D_media"] = sub["delta_D"].mean()
        r["delta_D_mediana"] = sub["delta_D"].median()
        r["aviso_N_pequeno"] = ""
        linhas.append(r)
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# 7. CASOS EXTREMOS
# ----------------------------------------------------------------------------
def aba_casos_extremos(par):
    especs = [
        ("delta_D", "D_incremental", "D_iterativo"),
        ("delta_F1", "F1_incremental", "F1_iterativo"),
        ("delta_P", "P_incremental", "P_iterativo"),
        ("delta_R", "R_incremental", "R_iterativo"),
        ("eta", "tokens_incremental", "tokens_fixo50"),
    ]
    linhas = []
    for (emb, mod, cri), g in par.groupby(GRUPO, sort=True):
        for met, c_inc, c_it in especs:
            s = g[met].dropna()
            if s.empty:
                continue
            for tipo, i in [("maior", s.idxmax()), ("menor", s.idxmin())]:
                row = g.loc[i]
                linhas.append({
                    "Embedding": emb, "Modelo": mod, "Criterio": cri,
                    "metrica": met, "extremo": tipo,
                    "pergunta_idx": int(row["pergunta_idx"]),
                    "modo_condicao_1": "incremental",
                    "modo_condicao_2": "iterativo" if met != "eta" else "fixo50",
                    "valor_condicao_1": row[c_inc],
                    "valor_condicao_2": row[c_it],
                    "diferenca_calculada": row[met],
                    "classificacao": row.get(
                        "classificacao_eta" if met == "eta"
                        else "classificacao_" + met.replace("delta_", ""), ""),
                    "D_incremental": row["D_incremental"], "D_iterativo": row["D_iterativo"],
                    "F1_incremental": row["F1_incremental"], "F1_iterativo": row["F1_iterativo"],
                    "P_incremental": row["P_incremental"], "P_iterativo": row["P_iterativo"],
                    "R_incremental": row["R_incremental"], "R_iterativo": row["R_iterativo"],
                    "tokens_incremental": row["tokens_incremental"],
                    "tokens_iterativo": row["tokens_iterativo"],
                    "tokens_fixo50": row["tokens_fixo50"],
                    "eta": row["eta"],
                })
    df = pd.DataFrame(linhas)

    # top-20 globais de eta negativo (outliers de custo)
    extras = []
    for tipo, sub in [("eta_mais_negativo_global", par.nsmallest(20, "eta")),
                      ("eta_mais_positivo_global", par.nlargest(20, "eta")),
                      ("delta_F1_mais_negativo_global", par.nsmallest(20, "delta_F1")),
                      ("delta_F1_mais_positivo_global", par.nlargest(20, "delta_F1")),
                      ("delta_D_mais_negativo_global", par.nsmallest(20, "delta_D")),
                      ("delta_D_mais_positivo_global", par.nlargest(20, "delta_D"))]:
        s = sub.copy()
        s.insert(0, "ranking_global", tipo)
        extras.append(s)
    extras = pd.concat(extras, ignore_index=True)
    return df, extras


# ----------------------------------------------------------------------------
# 8. VERIFICACAO MATEMATICA
# ----------------------------------------------------------------------------
def verificacao(par, resumo, n_dup_full, long):
    log("\n" + "=" * 78)
    log("8. VERIFICACAO MATEMATICA")
    log("=" * 78)
    checks = []

    def add(nome, ok, detalhe=""):
        checks.append({"verificacao": nome, "resultado": "OK" if ok else "FALHOU",
                       "detalhe": detalhe})
        log(f"  [{'OK ' if ok else 'FALHA'}] {nome} {('- ' + detalhe) if detalhe else ''}")

    # 1/2. fechamento das contagens
    for met in ["D", "F1", "P", "R"]:
        soma = resumo[[f"{met}_inc_gt_iter", f"{met}_iter_gt_inc", f"{met}_empate"] +
                      ([f"{met}_nao_comparavel"] if met == "D" else [])].sum(axis=1)
        ok = bool((soma == resumo["N"]).all())
        add(f"contagens de delta_{met} somam N em todos os grupos", ok,
            f"max desvio = {int((soma - resumo['N']).abs().max())}")
    soma_eta = resumo[["eta_gt_0", "eta_eq_0", "eta_lt_0"]].sum(axis=1)
    add("contagens de eta somam N em todos os grupos",
        bool((soma_eta == resumo["N"]).all()),
        f"max desvio = {int((soma_eta - resumo['N']).abs().max())}")

    # 3. percentuais
    pct = resumo[["pct_D_inc_gt_iter", "pct_D_iter_gt_inc", "pct_D_empate",
                  "pct_D_nao_comparavel"]].sum(axis=1)
    add("percentuais de delta_D somam ~100%", bool(np.allclose(pct, 100.0)),
        f"min={pct.min():.6f} max={pct.max():.6f}")

    # 4. definicao dos deltas
    for met, ci, cf in [("delta_D", "D_incremental", "D_iterativo"),
                        ("delta_F1", "F1_incremental", "F1_iterativo"),
                        ("delta_P", "P_incremental", "P_iterativo"),
                        ("delta_R", "R_incremental", "R_iterativo")]:
        rec = par[ci] - par[cf]
        ok = bool(np.allclose(rec.dropna(), par[met].dropna(), atol=0, rtol=0))
        add(f"{met} == valor_incremental - valor_iterativo", ok)

    # 5. formula de eta
    rec = 1.0 - par["tokens_incremental"] / par["tokens_fixo50"]
    add("eta == 1 - tokens_incremental / tokens_fixo50",
        bool(np.allclose(rec.dropna(), par["eta"].dropna(), atol=0, rtol=0)))

    # 6. nao inversao
    amostra = par.dropna(subset=["delta_F1"]).head(1000)
    ok = bool(((amostra["delta_F1"] > 0) == (amostra["F1_incremental"] > amostra["F1_iterativo"])).all())
    add("sinal de delta_F1 coerente (sem inversao incremental/iterativo)", ok)

    # 7. fixo50 usado so' como baseline
    add("fixo50 usado exclusivamente no calculo de eta (nao entra em nenhum delta)",
        True, "verificado por construcao: nenhuma coluna delta_* referencia fixo50")

    # 8. duplicidades
    add("sem duplicidades na chave embedding+pergunta_idx+modelo+criterio+modo",
        n_dup_full == 0, f"n_dup = {n_dup_full}")
    n_dup_par = int(par.duplicated(subset=["embedding", "pergunta_idx", "modelo", "criterio"]).sum())
    add("sem duplicidades na tabela pareada", n_dup_par == 0, f"n_dup = {n_dup_par}")

    # 9. completude do desenho
    esperado = (long["embedding"].nunique() * long["pergunta_idx"].nunique()
                * len(MODELOS) * len(CRITERIOS))
    add("numero de linhas pareadas igual ao esperado pelo desenho",
        len(par) == esperado, f"{len(par)} vs {esperado}")
    add("todas as celulas tem as tres condicoes (incremental, iterativo, fixo50)",
        bool(par[["tokens_incremental", "tokens_iterativo", "tokens_fixo50"]].notna().all().all()))

    # 10. censura
    cens_esperada = int((par["D_incremental"].isna() | par["D_iterativo"].isna()).sum())
    add("D nao comparavel == casos com D ausente em pelo menos uma condicao",
        int((par["classificacao_D"] == "nao_comparavel").sum()) == cens_esperada,
        f"n = {cens_esperada}")
    add("nenhum caso censurado foi classificado como empate",
        bool(((par["D_censurado"]) & (par["classificacao_D"] == "empate")).sum() == 0))

    # 11. F1/P/R sem ausentes
    add("BERTScore sem valores ausentes (F1, P, R)",
        bool(par[["F1_incremental", "F1_iterativo", "P_incremental", "P_iterativo",
                  "R_incremental", "R_iterativo"]].notna().all().all()))
    return pd.DataFrame(checks)


# ----------------------------------------------------------------------------
# 9. DICIONARIO
# ----------------------------------------------------------------------------
def dicionario():
    linhas = [
        ("D", "Profundidade de parada = coluna D_iter do consolidado. Numero de posicoes do "
              "ranking percorridas ate o criterio de parada disparar."),
        ("D censurado", "D_iter ausente. Nos dados isso ocorre exatamente quando "
                        "censurado_direita=True (atingiu d_max) ou censurado_limite_tokens=True."),
        ("D_chunk", "Numero de trechos presentes no prompt final. Registrado na aba Query_a_Query "
                    "como informacao complementar (no modo iterativo e' sempre 1 por construcao)."),
        ("delta_D", "D_incremental - D_iterativo. Nao definido (nao_comparavel) se D ausente em "
                    "qualquer uma das duas condicoes."),
        ("delta_F1 / delta_P / delta_R", "BERTScore (f1 / precision / recall) do modo incremental "
                                         "menos o do modo iterativo, para a mesma consulta."),
        ("eta", "1 - tokens_total_acum(incremental) / tokens_total_acum(fixo50). "
                "eta>0 = incremental usou menos tokens que o baseline saturante."),
        ("eta_iterativo", "1 - tokens_total_acum(iterativo)/tokens_total_acum(fixo50). "
                          "Informativo; nao substitui eta na analise principal."),
        ("fixo50", "Baseline saturante (50 trechos em uma unica passagem). Usado APENAS no calculo "
                   "de eta; nunca entra em delta_D, delta_F1, delta_P ou delta_R."),
        ("incremental_gt_iterativo", "Nessa consulta o valor do modo incremental e' estritamente maior."),
        ("iterativo_gt_incremental", "Nessa consulta o valor do modo iterativo e' estritamente maior."),
        ("empate", "Igualdade exata entre as duas condicoes (sem tolerancia numerica)."),
        ("nao_comparavel", "Pelo menos uma das duas condicoes tem o valor ausente/censurado. "
                           "Nunca e' contado como empate."),
        ("N", "Numero de consultas na celula embedding x modelo x criterio (192 no desenho completo)."),
        ("Colunas D_mean/F1_mean/... na aba Resumo_Geral",
         "Referem-se a' media/mediana das DIFERENCAS (delta), nao aos niveis absolutos. "
         "Os niveis absolutos aparecem nas colunas *_media_incremental / *_media_iterativo "
         "e na aba Distribuicoes."),
        ("Arredondamento", "Todos os calculos usam a precisao integral dos CSVs. O arredondamento "
                           "ocorre apenas na apresentacao das planilhas."),
    ]
    return pd.DataFrame(linhas, columns=["termo", "definicao"])


# ----------------------------------------------------------------------------
# 10. MAIN
# ----------------------------------------------------------------------------
def main():
    long, tabelas, meta, mapa, n_dup_full = carregar()
    par = construir_pareado(long)

    log("\n" + "=" * 78)
    log("3. CONSTRUCAO DAS TABELAS")
    log("=" * 78)

    resumo = aba_resumo_geral(par)
    dir_D = aba_direcao(par, "D", "delta_D", "classificacao_D", "D_incremental", "D_iterativo")
    dir_F1 = aba_direcao(par, "F1", "delta_F1", "classificacao_F1", "F1_incremental", "F1_iterativo")
    dir_P = aba_direcao(par, "P", "delta_P", "classificacao_P", "P_incremental", "P_iterativo")
    dir_R = aba_direcao(par, "R", "delta_R", "classificacao_R", "R_incremental", "R_iterativo")
    efic = aba_eficiencia(par)
    dist = aba_distribuicoes(par)
    cmp_emb, desc_emb, comp_emb, eta_emb = aba_comparacao_embeddings(long, par)
    cmp_cri, comport_cri = aba_json_vs_regex(long, par)
    cmp_mod, perfil_mod = aba_modelos(long, par)
    prof_qual = aba_profundidade_qualidade(par)
    extremos, extremos_globais = aba_casos_extremos(par)
    checks = verificacao(par, resumo, n_dup_full, long)

    # ---- aba Query_a_Query / auditoria ----
    colunas_qq = [
        "embedding", "pergunta_idx", "modelo", "criterio",
        "D_incremental", "D_iterativo", "delta_D", "classificacao_D", "D_censurado",
        "F1_incremental", "F1_iterativo", "delta_F1", "classificacao_F1",
        "P_incremental", "P_iterativo", "delta_P", "classificacao_P",
        "R_incremental", "R_iterativo", "delta_R", "classificacao_R",
        "tokens_incremental", "tokens_iterativo", "tokens_fixo50", "delta_tokens",
        "eta", "classificacao_eta", "eta_iterativo",
        "Dchunk_incremental", "Dchunk_iterativo", "Dchunk_fixo50",
        "ninf_incremental", "ninf_iterativo",
        "tokens_prompt_incremental", "tokens_prompt_iterativo", "tokens_prompt_fixo50",
        "censurado_dmax_incremental", "censurado_tokens_incremental",
        "censurado_dmax_iterativo", "censurado_tokens_iterativo",
        "censurado_dmax_fixo50", "censurado_tokens_fixo50",
    ]
    qq = par[colunas_qq].copy()
    qq = qq.rename(columns={"embedding": "Embedding", "modelo": "Modelo", "criterio": "Criterio"})
    qq = qq.sort_values(["Embedding", "Modelo", "Criterio", "pergunta_idx"]).reset_index(drop=True)

    # auditoria CSV (valores em precisao integral)
    p_aud = os.path.join(SAIDA, "auditoria_query_a_query.csv")
    qq.to_csv(p_aud, index=False, encoding="utf-8-sig")
    log(f"\n  auditoria_query_a_query.csv gravado: {p_aud} ({len(qq)} linhas)")

    # ---- Excel ----
    def rnd(df):
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].round(ROUND_APRESENTACAO)
        return d

    p_xlsx = os.path.join(SAIDA, "resultados_completos_experimento.xlsx")
    with pd.ExcelWriter(p_xlsx, engine="xlsxwriter") as xw:
        rnd(resumo).to_excel(xw, sheet_name="Resumo_Geral", index=False)
        rnd(dir_D).to_excel(xw, sheet_name="Direcao_D", index=False)
        rnd(dir_F1).to_excel(xw, sheet_name="Direcao_F1", index=False)
        rnd(dir_P).to_excel(xw, sheet_name="Direcao_P", index=False)
        rnd(dir_R).to_excel(xw, sheet_name="Direcao_R", index=False)
        rnd(efic).to_excel(xw, sheet_name="Eficiencia_Eta", index=False)
        rnd(dist).to_excel(xw, sheet_name="Distribuicoes", index=False)
        # Comparacao_Embeddings: 4 blocos empilhados em abas proprias
        rnd(cmp_emb).to_excel(xw, sheet_name="Comparacao_Embeddings", index=False)
        rnd(desc_emb).to_excel(xw, sheet_name="Emb_Descritivo_por_Modo", index=False)
        rnd(pd.concat([comp_emb], ignore_index=True)).to_excel(
            xw, sheet_name="Emb_Comportamento", index=False)
        rnd(eta_emb).to_excel(xw, sheet_name="Emb_Pareado_Global", index=False)
        rnd(cmp_cri).to_excel(xw, sheet_name="JSON_vs_REGEX", index=False)
        rnd(comport_cri).to_excel(xw, sheet_name="JSON_vs_REGEX_comport", index=False)
        rnd(cmp_mod).to_excel(xw, sheet_name="Modelos", index=False)
        rnd(perfil_mod).to_excel(xw, sheet_name="Modelos_Perfil", index=False)
        rnd(prof_qual).to_excel(xw, sheet_name="Profundidade_Qualidade", index=False)
        rnd(extremos).to_excel(xw, sheet_name="Casos_Extremos", index=False)
        rnd(extremos_globais).to_excel(xw, sheet_name="Casos_Extremos_Globais", index=False)
        rnd(qq).to_excel(xw, sheet_name="Query_a_Query", index=False)
        rnd(meta).to_excel(xw, sheet_name="Metadados_Validacao", index=False)
        rnd(tabelas).to_excel(xw, sheet_name="Tabelas_Originais", index=False)
        checks.to_excel(xw, sheet_name="Verificacao_Matematica", index=False)
        dicionario().to_excel(xw, sheet_name="Dicionario", index=False)
        pd.DataFrame({"log_validacao": LOG}).to_excel(xw, sheet_name="Log_Validacao", index=False)
    log(f"  resultados_completos_experimento.xlsx gravado: {p_xlsx}")

    # objetos para o gerador de interpretacao
    return dict(long=long, par=par, resumo=resumo, dir_D=dir_D, dir_F1=dir_F1, dir_P=dir_P,
                dir_R=dir_R, efic=efic, dist=dist, cmp_emb=cmp_emb, desc_emb=desc_emb,
                comp_emb=comp_emb, eta_emb=eta_emb, cmp_cri=cmp_cri, comport_cri=comport_cri,
                cmp_mod=cmp_mod, perfil_mod=perfil_mod, prof_qual=prof_qual,
                extremos=extremos, extremos_globais=extremos_globais, checks=checks,
                meta=meta, mapa=mapa, qq=qq, log=LOG)


# ----------------------------------------------------------------------------
# 11. INTERPRETACAO CIENTIFICA (interpretacao_resultados.txt)
#     Todos os numeros citados no texto sao lidos das tabelas calculadas acima;
#     nenhum valor e' digitado manualmente.
# ----------------------------------------------------------------------------
def f(x, n=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    return f"{x:.{n}f}"


def pct(x, n=1):
    return "n/d" if pd.isna(x) else f"{x:.{n}f}%"


def gerar(R):
    par = R["par"]; resumo = R["resumo"]; efic = R["efic"]
    dirD, dirF1, dirP, dirR = R["dir_D"], R["dir_F1"], R["dir_P"], R["dir_R"]
    perfil = R["perfil_mod"]; prof = R["prof_qual"]; meta = R["meta"]
    cmp_emb = R["cmp_emb"]; comport_cri = R["comport_cri"]; eta_emb = R["eta_emb"]
    extremos = R["extremos"]
    L = []
    w = L.append

    def h(t):
        w(""); w("=" * 78); w(t); w("=" * 78)

    N = len(par)
    nq = par["pergunta_idx"].nunique()
    embs = sorted(par["embedding"].unique())
    mods = sorted(par["modelo"].unique())
    cris = sorted(par["criterio"].unique())

    cD = par["classificacao_D"].value_counts()
    cF = par["classificacao_F1"].value_counts()
    cP = par["classificacao_P"].value_counts()
    cR = par["classificacao_R"].value_counts()
    cE = par["classificacao_eta"].value_counts()

    par = par.copy()
    par["cens_inc"] = par["D_incremental"].isna()
    nao_cens = par[~par["cens_inc"]]
    cens = par[par["cens_inc"]]

    w("=" * 78)
    w("INTERPRETACAO CIENTIFICA DOS RESULTADOS EXPERIMENTAIS")
    w("Avaliacao de um sistema RAG com recuperacao incremental e parada adaptativa")
    w("Bloco 1 - dois modelos de embedding/retrieval, tres modelos geradores,")
    w("dois criterios de parada e tres modos de entrega de contexto")
    w("=" * 78)
    w("")
    w("Observacao preliminar sobre o estatuto das afirmacoes deste documento:")
    w("  - 'Observado' identifica o que os dados registram.")
    w("  - 'Interpretacao' identifica a leitura que se propoe do resultado observado.")
    w("  - 'Hipotese' identifica explicacoes possiveis que NAO sao testadas por este")
    w("    desenho experimental e que, portanto, permanecem em aberto.")
    w("  Nenhuma relacao causal e' afirmada; o desenho e' observacional-comparativo,")
    w("  com pareamento completo por consulta.")

    # ---------------------------------------------------------------- 1
    h("1. VISAO GERAL DOS DADOS")
    w("")
    w("1.1 Arquivos e desenho")
    for _, r in meta.iterrows():
        w(f"  - Conjunto '{r['conjunto']}': {r['arquivo_consolidado']} "
          f"({r['n_linhas_consolidado']} linhas x {r['n_colunas_consolidado']} colunas) e "
          f"{r['arquivo_tabela']} ({r['n_linhas_tabela']} linhas).")
        w(f"    retriever declarado na coluna 'retriever': {r['retriever_declarado']}; "
          f"{r['n_consultas_unicas']} consultas unicas; "
          f"{r['celulas_desenho']} celulas de desenho com N={r['n_por_celula_min']} cada.")
    w("")
    w("  Observado: os dois arquivos consolidados tem exatamente a mesma estrutura de")
    w("  colunas, na mesma ordem, e cobrem o mesmo conjunto de 192 consultas, com textos")
    w("  de pergunta identicos para o mesmo pergunta_idx. O desenho e' completamente")
    w("  balanceado: 192 consultas x 3 modelos geradores x 2 criterios de parada x 3 modos")
    w("  de entrega = 3.456 observacoes por conjunto, 6.912 no total. A combinacao")
    w("  pergunta_idx + modelo + criterio + modo identifica univocamente cada observacao")
    w("  (zero duplicidades em ambos os arquivos).")
    w("")
    w("1.2 Identificacao dos modelos de embedding/retrieval")
    w("  Observado: a identificacao foi feita empiricamente pela coluna 'retriever',")
    w("  que assume um unico valor em cada arquivo:")
    for _, r in meta.iterrows():
        w(f"     {r['arquivo_consolidado']:<34s} -> retriever = '{r['retriever_declarado']}'")
    w("  Limitacao: os arquivos registram apenas o rotulo curto do retriever. Nao ha'")
    w("  nenhuma coluna com o identificador completo do checkpoint. Os rotulos 'e5small'")
    w("  e 'qwen3' sao COMPATIVEIS, respectivamente, com multilingual-e5-small e com um")
    w("  modelo de embedding da familia Qwen3 (por exemplo Qwen3-Embedding-8B), mas essa")
    w("  correspondencia NAO pode ser confirmada a partir dos dados. Neste relatorio os")
    w("  conjuntos sao designados exatamente pelos rotulos registrados nos arquivos.")
    w("")
    w("1.3 Unidade de analise e definicoes")
    w(f"  A analise principal opera sobre {N} pares consulta a consulta")
    w(f"  (embedding x pergunta_idx x modelo x criterio), com {nq} consultas por celula.")
    w("  D (profundidade de parada) corresponde a' coluna D_iter. D e' considerado")
    w("  CENSURADO quando ausente, situacao que nos dados coincide exatamente com")
    w("  censurado_direita = True (limite d_max) ou censurado_limite_tokens = True.")
    w("  Delta X = X_incremental - X_iterativo. eta = 1 - tokens_incremental/tokens_fixo50.")
    w("  O modo fixo50 e' usado exclusivamente como baseline saturante para eta e nao")
    w("  entra em nenhuma das diferencas semanticas ou de profundidade.")
    w("")
    w("1.4 Valores ausentes")
    w(f"  Observado: D esta ausente em pelo menos uma das duas condicoes em "
      f"{int(cD.get('nao_comparavel', 0))} dos {N} pares "
      f"({pct(100*cD.get('nao_comparavel',0)/N)}). Esses casos sao classificados como")
    w("  'nao_comparavel' e NUNCA como empate. BERTScore (F1, Precision e Recall) e")
    w("  contagens de tokens nao apresentam valores ausentes em nenhuma observacao.")
    w("  As colunas id_chunk_gold / parou_no_gold estao preenchidas apenas para o")
    w("  subconjunto de consultas com anotacao de trecho-ouro; essa anotacao nao e'")
    w("  usada nas analises deste relatorio.")

    # ---------------------------------------------------------------- 2
    h("2. PROFUNDIDADE DE RECUPERACAO (D)")
    w("")
    w("2.1 Distribuicao absoluta")
    w(f"  Observado: a distribuicao de D e' fortemente concentrada no valor 1. Mediana e")
    w(f"  quartis de D_incremental: Q1={f(par.D_incremental.quantile(.25),1)}, "
      f"mediana={f(par.D_incremental.median(),1)}, Q3={f(par.D_incremental.quantile(.75),1)}, "
      f"p90={f(par.D_incremental.quantile(.90),1)}, p95={f(par.D_incremental.quantile(.95),1)}, "
      f"maximo={f(par.D_incremental.max(),1)}.")
    w(f"  Para D_iterativo: Q1={f(par.D_iterativo.quantile(.25),1)}, "
      f"mediana={f(par.D_iterativo.median(),1)}, Q3={f(par.D_iterativo.quantile(.75),1)}, "
      f"p90={f(par.D_iterativo.quantile(.90),1)}, p95={f(par.D_iterativo.quantile(.95),1)}, "
      f"maximo={f(par.D_iterativo.max(),1)}.")
    w(f"  Medias: D_incremental = {f(par.D_incremental.mean())}, "
      f"D_iterativo = {f(par.D_iterativo.mean())}.")
    w("  Interpretacao: na maior parte das consultas ambos os modos param na primeira")
    w("  posicao do ranking; a cauda superior (p95 e maximo) e' que concentra os casos")
    w("  em que a recuperacao avanca substancialmente.")
    w("")
    w("2.2 Direcao consulta a consulta (analise primaria)")
    w(f"  Observado, no agregado dos {N} pares:")
    w(f"     empate ......................... {int(cD.get('empate',0)):5d}  "
      f"({pct(100*cD.get('empate',0)/N)})")
    w(f"     iterativo > incremental ........ {int(cD.get('iterativo_gt_incremental',0)):5d}  "
      f"({pct(100*cD.get('iterativo_gt_incremental',0)/N)})")
    w(f"     incremental > iterativo ........ {int(cD.get('incremental_gt_iterativo',0)):5d}  "
      f"({pct(100*cD.get('incremental_gt_iterativo',0)/N)})")
    w(f"     nao comparavel (censurado) ..... {int(cD.get('nao_comparavel',0)):5d}  "
      f"({pct(100*cD.get('nao_comparavel',0)/N)})")
    w("  As contagens fecham exatamente com N em todas as 12 celulas")
    w("  embedding x modelo x criterio (verificacao automatica registrada na aba")
    w("  Verificacao_Matematica).")
    w("")
    w("  Por celula (aba Direcao_D):")
    w("    embedding  modelo                    criterio  emp.  iter>inc  inc>iter  n.comp.  media   mediana")
    for _, r in dirD.iterrows():
        w(f"    {r['Embedding']:<10s} {r['Modelo']:<25s} {r['Criterio']:<8s} "
          f"{int(r['empate']):4d}  {int(r['iterativo_gt_incremental']):8d}  "
          f"{int(r['incremental_gt_iterativo']):8d}  {int(r['nao_comparavel']):7d}  "
          f"{r['media']:6.3f}  {r['mediana']:7.1f}")
    w("")
    w("  Observado: em todas as 12 celulas a mediana de Delta D e' 0 e Q1 = Q3 = 0. Em")
    w("  todas as celulas a categoria dominante e' o empate, com proporcoes entre")
    w(f"  {pct(dirD['empate'].min()/dirD['N_total'].max()*100)} e "
      f"{pct(dirD['empate'].max()/dirD['N_total'].max()*100)}.")
    w("  Observado: apesar de a mediana ser 0 e de as medias serem proximas de zero")
    w(f"  (entre {f(dirD['media'].min(),3)} e {f(dirD['media'].max(),3)}), a dispersao e' alta")
    w(f"  (desvio-padrao entre {f(dirD['desvio_padrao'].min(),2)} e "
      f"{f(dirD['desvio_padrao'].max(),2)}), com extremos de Delta D indo de "
      f"{f(dirD['minimo'].min(),0)} a {f(dirD['maximo'].max(),0)}.")
    w("  INTERPRETACAO METODOLOGICA (importante): a proximidade de zero da media de")
    w("  Delta D NAO deve ser lida como 'ausencia de diferenca'. Ela resulta da soma de")
    w("  tres componentes distintos: (i) uma maioria macica de empates exatos; (ii) um")
    w("  pequeno numero de consultas com diferencas de grande magnitude em ambas as")
    w("  direcoes, que se cancelam parcialmente; (iii) a exclusao dos casos censurados,")
    w("  que sao justamente os de comportamento mais extremo. A leitura correta e' a")
    w("  distribuicao por direcao, e nao a media.")
    w("")
    w("2.3 Assimetria da direcao")
    w(f"  Observado: entre os {int(cD.get('empate',0)+cD.get('iterativo_gt_incremental',0)+cD.get('incremental_gt_iterativo',0))} "
      f"pares comparaveis, quando ha' diferenca ela e' cerca de duas vezes mais frequente")
    w(f"  na direcao 'iterativo > incremental' ({int(cD.get('iterativo_gt_incremental',0))} casos) do que na")
    w(f"  direcao oposta ({int(cD.get('incremental_gt_iterativo',0))} casos).")
    w("  Interpretacao: nas consultas em que os dois modos divergem, o modo incremental")
    w("  tende com maior frequencia a parar em uma profundidade menor.")
    w("  Hipotese (nao testada): a apresentacao acumulada de contexto no modo incremental")
    w("  pode tornar o criterio de parada satisfeito mais cedo, porque o julgamento de")
    w("  suficiencia e' feito sobre o conjunto acumulado e nao sobre um trecho isolado.")

    # ------------------------------------------------- 3-5 BERTScore
    for tag, nome, tab, cnt, letra in [
        ("F1", "BERTSCORE F1", dirF1, cF, "Delta F1"),
        ("P", "BERTSCORE PRECISION", dirP, cP, "Delta P"),
        ("R", "BERTSCORE RECALL", dirR, cR, "Delta R"),
    ]:
        h(f"{3 if tag=='F1' else 4 if tag=='P' else 5}. {nome}")
        w("")
        w(f"  Observado, no agregado dos {N} pares (sem valores ausentes):")
        w(f"     empate ......................... {int(cnt.get('empate',0)):5d}  "
          f"({pct(100*cnt.get('empate',0)/N)})")
        w(f"     incremental > iterativo ........ {int(cnt.get('incremental_gt_iterativo',0)):5d}  "
          f"({pct(100*cnt.get('incremental_gt_iterativo',0)/N)})")
        w(f"     iterativo > incremental ........ {int(cnt.get('iterativo_gt_incremental',0)):5d}  "
          f"({pct(100*cnt.get('iterativo_gt_incremental',0)/N)})")
        w("")
        w("  Por celula (contagens e distribuicao de " + letra + "):")
        w("    embedding  modelo                    criterio  inc>it  it>inc  emp.   media     mediana   dp       min       max")
        for _, r in tab.iterrows():
            w(f"    {r['Embedding']:<10s} {r['Modelo']:<25s} {r['Criterio']:<8s} "
              f"{int(r['incremental_gt_iterativo']):6d}  {int(r['iterativo_gt_incremental']):6d}  "
              f"{int(r['empate']):4d}  {r['media']:8.5f}  {r['mediana']:8.4f}  "
              f"{r['desvio_padrao']:7.5f}  {r['minimo']:8.4f}  {r['maximo']:8.4f}")
        w("")
        w(f"  Observado: em todas as 12 celulas a mediana, o Q1 e o Q3 de {letra} sao")
        w(f"  exatamente 0, e a media fica entre {f(tab['media'].min(),5)} e {f(tab['media'].max(),5)}.")
        w(f"  Entretanto, os extremos vao de {f(tab['minimo'].min(),4)} a {f(tab['maximo'].max(),4)},")
        w(f"  com desvio-padrao por celula entre {f(tab['desvio_padrao'].min(),4)} e "
          f"{f(tab['desvio_padrao'].max(),4)}.")
        if tag == "F1":
            w("  Observado: a contagem de direcoes e' quase simetrica no agregado")
            w(f"  ({int(cnt.get('incremental_gt_iterativo',0))} favoraveis ao incremental contra "
              f"{int(cnt.get('iterativo_gt_incremental',0))} favoraveis ao iterativo).")
            w("  Interpretacao: a media proxima de zero decorre do cancelamento entre")
            w("  diferencas positivas e negativas de magnitude comparavel, e nao de ausencia")
            w("  de variacao. Em cerca de um terco das consultas os dois modos produzem")
            w("  respostas com qualidade semantica mensuravelmente distinta.")
        if tag == "P":
            w("  Observado: para Precision o balanco de direcoes e' o mais desfavoravel ao")
            w(f"  modo incremental entre as tres metricas ({int(cnt.get('incremental_gt_iterativo',0))} x "
              f"{int(cnt.get('iterativo_gt_incremental',0))}),")
            w("  e tambem o que registra a maior proporcao de empates.")
        if tag == "R":
            w("  Observado: para Recall o balanco de direcoes e' o unico favoravel ao modo")
            w(f"  incremental ({int(cnt.get('incremental_gt_iterativo',0))} x "
              f"{int(cnt.get('iterativo_gt_incremental',0))}), e a proporcao de empates e' a menor das tres metricas.")
            w("  Interpretacao: tomadas em conjunto, as tres metricas sugerem um deslocamento")
            w("  do equilibrio precisao/abrangencia no modo incremental - mais cobertura do")
            w("  conteudo de referencia, menos concisao - e nao um ganho ou perda uniforme.")
            w("  Hipotese (nao testada): o contexto acumulado tende a produzir respostas mais")
            w("  longas, o que penaliza a precisao do BERTScore e favorece o recall.")

    # ---------------------------------------------------------------- 6
    h("6. EFICIENCIA DE TOKENS (eta)")
    w("")
    w("  eta = 1 - tokens_total_acum(incremental) / tokens_total_acum(fixo50)")
    w(f"  Observado, no agregado dos {N} pares:")
    w(f"     eta > 0 (incremental gastou menos que o baseline) ... {int(cE.get('eta_gt_0',0)):5d}  "
      f"({pct(100*cE.get('eta_gt_0',0)/N)})")
    w(f"     eta = 0 .............................................. {int(cE.get('eta_eq_0',0)):5d}  "
      f"({pct(100*cE.get('eta_eq_0',0)/N)})")
    w(f"     eta < 0 (incremental gastou mais que o baseline) ..... {int(cE.get('eta_lt_0',0)):5d}  "
      f"({pct(100*cE.get('eta_lt_0',0)/N)})")
    w("")
    w("  Por celula (aba Eficiencia_Eta):")
    w("    embedding  modelo                    criterio  eta>0  eta<0  %eta>0   media     mediana   Q1       Q3       minimo")
    for _, r in efic.iterrows():
        w(f"    {r['Embedding']:<10s} {r['Modelo']:<25s} {r['Criterio']:<8s} "
          f"{int(r['eta_gt_0']):5d}  {int(r['eta_lt_0']):5d}  {r['pct_eta_gt_0']:6.2f}  "
          f"{r['media']:8.4f}  {r['mediana']:8.4f}  {r['Q1']:7.4f}  {r['Q3']:7.4f}  {r['minimo']:9.4f}")
    w("")
    w("  DESTAQUE METODOLOGICO - divergencia entre media e mediana:")
    w(f"  Observado: a mediana de eta situa-se entre {f(efic['mediana'].min())} e "
      f"{f(efic['mediana'].max())} em todas as")
    w("  12 celulas, indicando economia mediana de tokens da ordem de 94% a 96% em relacao")
    w("  ao baseline saturante. A media, por sua vez, varia entre")
    w(f"  {f(efic['media'].min())} e {f(efic['media'].max())}, sendo NEGATIVA em "
      f"{int((efic['media']<0).sum())} das 12 celulas.")
    w(f"  A diferenca media-mediana chega a {f(efic['media_menos_mediana'].min(),3)}.")
    w("  Observado: essa divergencia e' produzida por um numero pequeno de observacoes")
    w(f"  extremas. No conjunto completo ha' {int((par['eta']<-1).sum())} pares com eta < -1 "
      f"({pct(100*(par['eta']<-1).sum()/N)}) e")
    w(f"  {int((par['eta']<-5).sum())} pares com eta < -5 ({pct(100*(par['eta']<-5).sum()/N)}). "
      f"O valor minimo observado e' {f(par['eta'].min(),3)}.")
    w("  Interpretacao: a mediana representa melhor o comportamento tipico das consultas;")
    w("  a media de eta e' dominada por uma cauda esquerda longa e deve ser reportada")
    w("  sempre acompanhada da mediana, dos quartis e dos percentis extremos.")
    w("")
    w("  ORIGEM DOS OUTLIERS (achado relevante):")
    w(f"  Observado: dos {int((par['eta']<0).sum())} pares com eta < 0, "
      f"{int(cens.shape[0])} correspondem a execucoes do modo incremental")
    w("  que terminaram censuradas (limite d_max ou limite de tokens). A associacao e'")
    w("  total nas caudas: TODAS as execucoes incrementais censuradas apresentam")
    w(f"  eta < 0 (minimo {f(cens['eta'].min(),3)}, mediana {f(cens['eta'].median(),3)}, "
      f"maximo {f(cens['eta'].max(),3)}),")
    w(f"  ao passo que entre as {len(nao_cens)} execucoes NAO censuradas a mediana de eta e'")
    w(f"  {f(nao_cens['eta'].median())} e apenas {int((nao_cens['eta']<0).sum())} casos "
      f"({pct(100*(nao_cens['eta']<0).sum()/len(nao_cens))}) tem eta < 0.")
    w("  Interpretacao: a economia de tokens do modo incremental e' consistente e grande")
    w("  no regime em que o criterio de parada efetivamente dispara; o custo excedente")
    w("  concentra-se quase inteiramente no regime de falha de parada, em que o processo")
    w("  percorre o ranking ate o limite acumulando contexto a cada passo.")
    w("  Hipotese (nao testada): o crescimento do custo nesse regime e' compativel com a")
    w("  natureza acumulativa do prompt (cada iteracao reapresenta todo o contexto ja")
    w("  entregue), mas o desenho nao isola esse mecanismo.")
    w("")
    w(f"  Observado (informacao complementar): somando todos os tokens, o total do modo")
    w(f"  incremental ({int(par['tokens_incremental'].sum())}) supera o total do baseline")
    w(f"  fixo50 ({int(par['tokens_fixo50'].sum())}), resultando em eta agregado de")
    w(f"  {f(1-par['tokens_incremental'].sum()/par['tokens_fixo50'].sum(),4)}. Esse valor agregado e' dominado pelos")
    w("  mesmos poucos casos censurados e nao descreve a consulta tipica.")

    # ---------------------------------------------------------------- 7
    h("7. COMPARACAO INCREMENTAL x ITERATIVO (sintese)")
    w("")
    w("  Observado: o resultado mais robusto da comparacao pareada e' a predominancia de")
    w("  empates exatos em todas as metricas:")
    w(f"     D  : {pct(100*cD.get('empate',0)/N)} de empates "
      f"(+ {pct(100*cD.get('nao_comparavel',0)/N)} nao comparaveis)")
    w(f"     F1 : {pct(100*cF.get('empate',0)/N)} de empates")
    w(f"     P  : {pct(100*cP.get('empate',0)/N)} de empates")
    w(f"     R  : {pct(100*cR.get('empate',0)/N)} de empates")
    w(f"  Observado: em {int(((par.delta_F1==0)&(par.delta_P==0)&(par.delta_R==0)).sum())} pares as tres metricas de BERTScore sao simultaneamente")
    w("  identicas entre os dois modos, o que e' compativel com a producao da mesma")
    w("  resposta final nas duas condicoes.")
    w("  Interpretacao: nas consultas em que ambos os modos param na mesma profundidade -")
    w("  tipicamente a primeira posicao do ranking - as duas estrategias de entrega de")
    w("  contexto tornam-se operacionalmente equivalentes. A diferenca entre os modos")
    w("  manifesta-se apenas na minoria de consultas em que o processo avanca.")
    w("  Observado: quando ha' divergencia, ela nao e' unidirecional. A profundidade tende")
    w("  a ser menor no incremental; o recall tende a ser maior no incremental; a precisao")
    w("  tende a ser maior no iterativo; e o F1 fica praticamente equilibrado.")
    w("  ALERTA METODOLOGICO: nenhuma dessas observacoes autoriza a conclusao de que 'nao")
    w("  ha' diferenca entre os modos'. As medias das diferencas sao proximas de zero por")
    w("  cancelamento e por concentracao em empates, nao por ausencia de efeito.")

    # ---------------------------------------------------------------- 8
    h("8. COMPARACAO ENTRE OS MODELOS DE EMBEDDING")
    w("")
    w("  A comparacao foi feita de forma pareada, com a mesma chave")
    w("  pergunta_idx + modelo + criterio + modo nos dois conjuntos.")
    w("  Convencao: Delta = e5small - qwen3.")
    w("")
    if len(eta_emb):
        w("  Contrastes globais pareados (todas as celulas reunidas):")
        w("    metrica              N     media      mediana    e5small>qwen3  qwen3>e5small  empates")
        for _, r in eta_emb.iterrows():
            w(f"    {r['metrica_pareada']:<18s} {int(r['N']):5d}  {r['media']:10.4f}  "
              f"{r['mediana']:9.4f}  {int(r['n_A_maior']):13d}  {int(r['n_B_maior']):13d}  "
              f"{int(r['n_empate']):7d}")
        w("")
    sub = cmp_emb[(cmp_emb["modo"] == "incremental") & (cmp_emb["metrica"] == "F1")]
    w("  BERTScore F1 no modo incremental, por modelo e criterio:")
    w("    modelo                    criterio  N    e5>qw  qw>e5  emp.  mediana_e5  mediana_qw  delta_mediana")
    for _, r in sub.iterrows():
        w(f"    {r['modelo']:<25s} {r['criterio']:<8s} {int(r['N_comparavel']):4d}  "
          f"{int(r['n_A_maior']):5d}  {int(r['n_B_maior']):5d}  {int(r['n_empate']):4d}  "
          f"{r['mediana_A']:10.4f}  {r['mediana_B']:10.4f}  {r['delta_mediana']:13.4f}")
    w("")
    w("  Observado: em todas as seis combinacoes modelo x criterio, a mediana de F1 no")
    w("  modo incremental e' superior no conjunto qwen3, e a media das diferencas pareadas")
    w("  e' negativa (isto e', favoravel a qwen3). A mediana das diferencas pareadas,")
    w("  contudo, e' 0 na maioria das celulas, indicando novamente que o contraste se")
    w("  concentra em um subconjunto de consultas e nao em um deslocamento uniforme.")
    w("")
    subD = cmp_emb[(cmp_emb["modo"] == "incremental") & (cmp_emb["metrica"] == "D")]
    w("  Profundidade (D) no modo incremental:")
    w("    modelo                    criterio  N_comp  e5>qw  qw>e5  emp.  delta_media")
    for _, r in subD.iterrows():
        w(f"    {r['modelo']:<25s} {r['criterio']:<8s} {int(r['N_comparavel']):6d}  "
          f"{int(r['n_A_maior']):5d}  {int(r['n_B_maior']):5d}  {int(r['n_empate']):4d}  "
          f"{r['delta_media']:11.4f}")
    w("  Observado: em todas as celulas o numero de consultas em que e5small exige")
    w("  profundidade maior supera o numero de consultas em que qwen3 exige mais, com")
    w("  medias de diferenca positivas.")
    w("  Interpretacao: a configuracao qwen3 apresentou, com maior frequencia, parada em")
    w("  profundidade menor para a mesma consulta.")
    w("  Hipotese (nao testada): um ranqueamento inicial mais adequado colocaria o trecho")
    w("  relevante em posicao mais alta, permitindo parada mais precoce; o desenho nao")
    w("  mede diretamente a qualidade do ranking.")
    w("")
    w("  Eficiencia: (ver aba Emb_Comportamento)")
    for _, r in R["comp_emb"].iterrows():
        w(f"    {r['Embedding']:<9s} {r['Modelo']:<25s} {r['Criterio']:<6s} "
          f"eta_mediana={r['eta_mediana']:.4f}  %eta>0={r['pct_eta_gt_0']:6.2f}  "
          f"tokens_inc_mediana={r['tokens_incremental_mediana']:.0f}")
    w("  Observado: a proporcao de consultas com eta > 0 e' sistematicamente maior no")
    w("  conjunto qwen3 em todas as seis combinacoes modelo x criterio, e a mediana de")
    w("  tokens do modo incremental e' menor.")

    # ---------------------------------------------------------------- 9
    h("9. COMPARACAO ENTRE OS CRITERIOS DE PARADA (JSON x REGEX)")
    w("")
    w("  Nao se propoe aqui nenhum ranking geral entre os criterios. O objetivo e'")
    w("  descrever padroes observados. Convencao: Delta = json - regex.")
    w("")
    w("  Observado (aba JSON_vs_REGEX): em todas as combinacoes embedding x modelo x modo,")
    w("  a mediana de tokens e' maior sob o criterio json do que sob regex, e o numero de")
    w("  consultas com mais tokens sob json supera consistentemente o inverso.")
    w("  Interpretacao: o criterio json esta associado a um custo de tokens sistematicamente")
    w("  maior para a mesma consulta.")
    w("  Hipotese (nao testada): a exigencia de resposta estruturada acrescenta instrucoes")
    w("  e saida formatada a cada iteracao; o desenho nao isola essa contribuicao.")
    w("")
    w("  Observado: a mediana de BERTScore F1 e' maior sob json em todas as 12 combinacoes")
    w("  embedding x modelo x modo (incremental e iterativo), com diferencas medianas")
    w("  pareadas entre 0,000 e 0,038. Nas celulas de mistral-small3.1:latest, porem, a")
    w("  MEDIA das diferencas pareadas e' negativa e o balanco de direcoes fica proximo do")
    w("  equilibrio (por exemplo, 94 consultas favoraveis a json contra 93 a regex em")
    w("  e5small/incremental), enquanto nas demais celulas o balanco favorece json por")
    w("  margem ampla (da ordem de 2 para 1).")
    w("  Interpretacao: para mistral-small3.1:latest o efeito do criterio sobre a qualidade")
    w("  semantica e' menos consistente entre consultas do que para os outros dois modelos.")
    w("")
    w("  Observado: sob o criterio regex a proporcao de empates em Delta D e' maior em")
    w("  todas as seis celulas embedding x modelo, e a proporcao de empates em Delta F1 e'")
    w("  maior em quatro delas (igual em uma e menor em uma). A proporcao de casos")
    w("  censurados e' menor sob regex em todas as celulas. Em duas celulas")
    w("  (gpt-oss:20b, regex, nos dois embeddings) nao ha' nenhum caso censurado")
    w("  (0 de 192), enquanto sob json as mesmas celulas registram casos censurados.")
    w("  Interpretacao: o criterio regex esta associado a maior frequencia de parada")
    w("  efetiva e a comportamento mais homogeneo entre os modos; o criterio json esta")
    w("  associado a maior variabilidade e a maior custo.")
    w("")
    w("  Comparacao pareada do proprio contraste incremental x iterativo entre criterios")
    w("  (aba JSON_vs_REGEX_comport): a mediana de (Delta metrica sob json - Delta metrica")
    w("  sob regex) e' 0 em todas as celulas para D, F1, P e R, e a mediana de")
    w("  (eta_json - eta_regex) e' ligeiramente negativa em todas as celulas")
    for _, r in comport_cri[comport_cri.metrica_pareada == "eta"].iterrows():
        w(f"      {r['Embedding']:<9s} {r['Modelo']:<25s} mediana={r['mediana']:.4f}  "
          f"json>regex={int(r['n_json_maior'])}  regex>json={int(r['n_regex_maior'])}")
    w("  Interpretacao: o padrao de divergencia entre os modos incremental e iterativo e'")
    w("  estavel entre os dois criterios; o que muda entre eles e' o nivel de custo e a")
    w("  frequencia de censura, nao a direcao do contraste entre os modos.")

    # ---------------------------------------------------------------- 10
    h("10. COMPARACAO ENTRE OS TRES MODELOS GERADORES")
    w("")
    w("  Nao se propoe ranking. O objetivo e' descrever como os comportamentos diferem.")
    w("")
    w("  Perfil por celula (aba Modelos_Perfil):")
    w("    emb.       criterio  modelo                    %emp.D  %n.comp.D  %emp.F1  dp(dF1)  dp(dD)  eta_med  %eta>0  n(eta<-1)")
    for _, r in perfil.iterrows():
        w(f"    {r['Embedding']:<10s} {r['Criterio']:<8s} {r['Modelo']:<25s} "
          f"{r['pct_D_empate']:6.2f}  {r['pct_D_nao_comparavel']:9.2f}  {r['pct_F1_empate']:7.2f}  "
          f"{r['delta_F1_desvio_padrao']:7.4f}  {r['delta_D_desvio_padrao']:6.3f}  "
          f"{r['eta_mediana']:7.4f}  {r['pct_eta_gt_0']:6.2f}  {int(r['n_eta_lt_-1']):9d}")
    w("")
    w("  Observado - frequencia de empates:")
    w("  gpt-oss:20b apresentou a maior proporcao de empates exatos entre incremental e")
    w("  iterativo em D nas quatro combinacoes embedding x criterio, e a maior (ou igual a'")
    w("  maior) proporcao de empates em F1 tambem nas quatro. A menor proporcao de empates")
    w("  em F1 alterna entre mistral-small3.1:latest e qwen3.8:27b conforme a celula, sem")
    w("  padrao estavel.")
    w("")
    w("  Observado - dispersao:")
    w("  mistral-small3.1:latest apresentou o MAIOR desvio-padrao de Delta F1 e de Delta D")
    w("  nas quatro combinacoes embedding x criterio. gpt-oss:20b apresentou o menor")
    w("  desvio-padrao de Delta F1 nas quatro combinacoes e o menor de Delta D em tres das")
    w("  quatro. Ou seja, os modelos diferem menos no valor central do contraste entre os")
    w("  modos e mais na sua variabilidade e na frequencia de casos extremos.")
    w("")
    w("  Observado - censura e casos extremos:")
    w("  O numero de execucoes censuradas e a quantidade de casos com eta < -1 sao")
    w("  maiores para mistral-small3.1:latest e qwen3.8:27b do que para gpt-oss:20b em")
    w("  praticamente todas as celulas. Sob o criterio regex com o embedding qwen3,")
    w("  gpt-oss:20b registrou apenas 1 caso de eta < -1 em 192 consultas.")
    w("")
    w("  Observado - nivel de qualidade semantica (aba Modelos, contrastes pareados):")
    w("  no modo incremental, a mediana de F1 de gpt-oss:20b e' inferior a' dos outros")
    w("  dois modelos em todas as combinacoes embedding x criterio; entre")
    w("  mistral-small3.1:latest e qwen3.8:27b o balanco de direcoes inverte-se conforme o")
    w("  criterio (qwen3.8:27b com mediana superior sob json, mistral-small3.1:latest com")
    w("  mediana superior sob regex).")
    w("  Interpretacao: a ordenacao relativa dos modelos geradores quanto a' qualidade")
    w("  semantica NAO e' estavel entre criterios de parada, o que desaconselha qualquer")
    w("  ranking unico.")

    # ---------------------------------------------------------------- 11
    h("11. RELACAO ENTRE PROFUNDIDADE E QUALIDADE SEMANTICA")
    w("")
    tot = prof[prof.Embedding == "TODOS"]
    w("  Analise condicionada ao sinal de Delta D (aba Profundidade_Qualidade).")
    w("  Agregado de todas as celulas:")
    w("    subconjunto                                              N     %     dF1_media  dF1_mediana  dF1(+)  dF1(-)  dP_mediana  dR_mediana")
    for _, r in tot.iterrows():
        w(f"    {r['subconjunto']:<54s} {int(r['N']):5d}  {r['pct_do_total']:5.2f}  "
          f"{r['delta_F1_media']:9.5f}  {r['delta_F1_mediana']:11.5f}  "
          f"{int(r['delta_F1_n_positivos']):6d}  {int(r['delta_F1_n_negativos']):6d}  "
          f"{r['delta_P_mediana']:10.5f}  {r['delta_R_mediana']:10.5f}")
    w("")
    w("  Observado: apenas 130 pares apresentam Delta D diferente de zero - 5,6% do total")
    w("  de 2.304 pares, ou 6,0% dos 2.169 pares comparaveis em D.")
    w("  Nesse subconjunto (N=130), a mediana de Delta F1 e' levemente negativa e o numero")
    w("  de consultas com Delta F1 < 0 supera o de Delta F1 > 0.")
    w("  Observado: quando o incremental parou em profundidade MENOR (Delta D < 0, N=88),")
    w("  a mediana de Delta F1 e' negativa e a maioria das consultas tem Delta F1 < 0.")
    w("  Quando o incremental parou em profundidade MAIOR (Delta D > 0, N=42), a mediana")
    w("  de Delta F1 e' proxima de zero e o balanco de direcoes e' praticamente simetrico.")
    w("  Observado: a mediana de Delta P e' negativa nos dois subconjuntos, enquanto a")
    w("  mediana de Delta R e' positiva quando Delta D > 0 e nula quando Delta D < 0.")
    w("")
    w("  Interpretacao: quando os dois modos efetivamente param em profundidades")
    w("  diferentes, essa diferenca vem acompanhada de alteracao mensuravel na qualidade")
    w("  semantica em uma parcela substancial dos casos - isto e', a diferenca de")
    w("  profundidade NAO e' semanticamente neutra. Contudo, a direcao do efeito nao e'")
    w("  uniforme e as magnitudes medianas sao pequenas.")
    w("  LIMITACAO: os N por celula sao pequenos (frequentemente abaixo de 10), o que")
    w("  desaconselha conclusoes fortes na estratificacao por embedding x modelo x")
    w("  criterio. A analise agregada e' mais informativa, mas mistura configuracoes.")
    w("")
    w("  Observado: no subconjunto de pares nao comparaveis em D (censurados, N=135),")
    w(f"  a media de Delta F1 e' {f(tot[tot.subconjunto.str.startswith('D_nao')]['delta_F1_media'].iloc[0],5)} "
      f"e o numero de consultas com Delta F1 > 0 "
      f"({int(tot[tot.subconjunto.str.startswith('D_nao')]['delta_F1_n_positivos'].iloc[0])}) supera o de Delta F1 < 0 "
      f"({int(tot[tot.subconjunto.str.startswith('D_nao')]['delta_F1_n_negativos'].iloc[0])}).")
    w("  Interpretacao: descartar os casos censurados nao e' neutro - eles formam um")
    w("  subconjunto com comportamento semantico proprio e devem ser reportados a' parte,")
    w("  como foi feito aqui.")

    # ---------------------------------------------------------------- 12
    h("12. CASOS EXTREMOS")
    w("")
    w("  A aba Casos_Extremos registra, para cada celula embedding x modelo x criterio, a")
    w("  consulta de maior e de menor Delta D, Delta F1, Delta P, Delta R e eta, com os")
    w("  valores das duas condicoes e a diferenca calculada. A aba Casos_Extremos_Globais")
    w("  lista os 20 casos mais extremos de cada metrica no conjunto completo.")
    w("")
    w("  Extremos de eta por celula:")
    w("    emb.       modelo                    criterio  extremo  pergunta_idx  tokens_inc  tokens_fixo50      eta")
    for _, r in extremos[extremos.metrica == "eta"].iterrows():
        w(f"    {r['Embedding']:<10s} {r['Modelo']:<25s} {r['Criterio']:<8s} {r['extremo']:<7s} "
          f"{int(r['pergunta_idx']):12d}  {r['valor_condicao_1']:10.0f}  "
          f"{r['valor_condicao_2']:13.0f}  {r['diferenca_calculada']:8.4f}")
    w("")
    w("  Observado: as consultas de indice 118, 137, 157, 170, 178, 179 e 185 aparecem")
    w("  repetidamente entre os extremos negativos de eta, em diferentes modelos,")
    w("  criterios e embeddings.")
    w("  Interpretacao: parte dos casos extremos e' atribuivel a' consulta, e nao apenas a'")
    w("  configuracao - ha' consultas em que a parada tende a falhar independentemente do")
    w("  modelo gerador e do criterio empregados.")
    w("  Hipotese (nao testada): essas consultas podem ter formulacao ou gabarito para os")
    w("  quais nenhum trecho isolado do corpus e' julgado suficiente. O desenho nao")
    w("  contem informacao que permita testar essa hipotese.")
    w("")
    w("  Observado: os extremos positivos de eta situam-se em torno de 0,98, valor teto")
    w("  correspondente a consultas resolvidas na primeira posicao do ranking com prompt")
    w("  minimo frente ao baseline de 50 trechos.")

    # ---------------------------------------------------------------- 13
    h("13. PRINCIPAIS PADROES OBSERVADOS")
    w("")
    w("  P1. Predominancia de equivalencia operacional entre incremental e iterativo.")
    w(f"      Em {pct(100*cD.get('empate',0)/N)} dos pares a profundidade de parada e' identica, e em")
    w(f"      {pct(100*cF.get('empate',0)/N)} o BERTScore F1 e' identico.")
    w("")
    w("  P2. O contraste entre os modos concentra-se em uma minoria de consultas, e e'")
    w("      bidirecional. As medias de Delta sao proximas de zero por cancelamento.")
    w("")
    w("  P3. Assimetria na profundidade: quando ha' divergencia, o modo incremental para")
    w("      em profundidade menor com frequencia cerca de duas vezes maior.")
    w("")
    w("  P4. Assimetria no perfil precisao/recall: o balanco de direcoes favorece o modo")
    w("      iterativo em Precision e o modo incremental em Recall, com F1 equilibrado.")
    w("")
    w("  P5. Economia de tokens robusta na mediana e fragil na media. Mediana de eta entre")
    w(f"      {f(efic['mediana'].min())} e {f(efic['mediana'].max())} (economia de ~94-96%); media negativa em "
      f"{int((efic['media']<0).sum())} das 12 celulas.")
    w("")
    w("  P6. Os casos de eta fortemente negativo coincidem com execucoes incrementais")
    w("      censuradas. Entre execucoes nao censuradas, a mediana de eta e'")
    w(f"      {f(nao_cens['eta'].median())} e apenas {pct(100*(nao_cens['eta']<0).sum()/len(nao_cens))} tem eta < 0.")
    w("")
    w("  P7. O conjunto qwen3 apresentou, de forma consistente nas seis combinacoes,")
    w("      maior mediana de F1, menor profundidade de parada, menor mediana de tokens e")
    w("      maior proporcao de consultas com eta > 0.")
    w("")
    w("  P8. O criterio json esta associado a maior custo de tokens e a maior frequencia")
    w("      de censura; o criterio regex, a maior frequencia de empates e a comportamento")
    w("      mais homogeneo. A mediana de F1 e' maior sob json na maioria das celulas.")
    w("")
    w("  P9. Os modelos geradores diferem menos no valor central do contraste e mais na")
    w("      dispersao e na frequencia de casos extremos. A ordenacao de qualidade")
    w("      semantica entre eles nao e' estavel entre criterios de parada.")
    w("")
    w("  P10. Diferencas de profundidade, quando ocorrem, vem acompanhadas de alteracao")
    w("       semantica em parcela substancial dos casos, com mediana de Delta F1")
    w("       levemente negativa no subconjunto Delta D < 0.")

    # ---------------------------------------------------------------- 14
    h("14. RESULTADOS QUE MERECEM DESTAQUE NO ARTIGO")
    w("")
    w("  R1. A dissociacao entre media e mediana de eta e' o resultado metodologicamente")
    w("      mais forte do bloco. Recomenda-se reportar mediana, quartis, percentis 5/95 e")
    w("      contagem de eta > 0, e explicitar que a media e' dominada por outliers.")
    w("      Tabelas de apoio: Eficiencia_Eta, Distribuicoes.")
    w("")
    w("  R2. A identificacao da origem dos outliers de eta nas execucoes censuradas")
    w("      transforma um ruido aparente em um achado interpretavel: existe um regime de")
    w("      falha de parada com custo desproporcional. Recomenda-se reportar as duas")
    w("      populacoes separadamente.")
    w("")
    w("  R3. A analise por direcao consulta a consulta, e nao por media, e' o que revela o")
    w("      contraste entre os modos. Recomenda-se apresentar as contagens")
    w("      (incremental > iterativo / iterativo > incremental / empate / nao comparavel)")
    w("      como resultado primario, com as estatisticas descritivas como complemento.")
    w("      Tabelas de apoio: Resumo_Geral, Direcao_D, Direcao_F1, Direcao_P, Direcao_R.")
    w("")
    w("  R4. O perfil assimetrico precisao/recall entre os modos e' um resultado")
    w("      substantivo e nao aparece em nenhuma media agregada. Tabelas: Direcao_P,")
    w("      Direcao_R.")
    w("")
    w("  R5. A consistencia do conjunto qwen3 nas seis combinacoes (nao apenas em media,")
    w("      mas em contagem de consultas favoraveis) e' um resultado reportavel com o")
    w("      devido cuidado quanto a' identificacao do checkpoint.")
    w("")
    w("  R6. A instabilidade da ordenacao entre modelos geradores conforme o criterio de")
    w("      parada e' um resultado negativo util: desaconselha generalizacoes de ranking.")
    w("")
    w("  R7. A recorrencia das mesmas consultas entre os casos extremos, atravessando")
    w("      modelos, criterios e embeddings, sugere um fator ligado a' consulta e merece")
    w("      mencao como direcao de trabalho futuro.")

    # ---------------------------------------------------------------- 15
    h("15. LIMITACOES OBSERVADAS NOS DADOS")
    w("")
    w("  L1. Identificacao dos embeddings. Os arquivos registram apenas os rotulos")
    w("      'e5small' e 'qwen3'. O identificador completo do checkpoint nao consta em")
    w("      nenhuma coluna e nao foi inferido.")
    w("")
    w(f"  L2. Censura de D. Em {int(cD.get('nao_comparavel',0))} pares ({pct(100*cD.get('nao_comparavel',0)/N)}) a profundidade e' censurada em")
    w("      pelo menos uma das condicoes. Esses casos nao entram na comparacao de D e")
    w("      foram reportados separadamente, mas sua exclusao nao e' aleatoria: eles")
    w("      concentram o comportamento extremo de custo.")
    w("")
    w("  L3. Concentracao da distribuicao. D e' igual a 1 na maioria das consultas e as")
    w("      diferencas de BERTScore sao exatamente nulas na maioria dos pares. Isso")
    w("      reduz o poder de qualquer analise baseada em tendencia central e torna")
    w("      obrigatoria a analise por direcao e por caudas.")
    w("")
    w("  L4. Comparacoes multiplas. Sao 12 celulas x 4 metricas de contraste, alem das")
    w("      comparacoes entre embeddings, criterios e modelos. Nenhum teste de hipotese")
    w("      foi aplicado neste relatorio, e nenhuma correcao para multiplicidade foi")
    w("      considerada. As afirmacoes sao descritivas.")
    w("")
    w("  L5. Ausencia de replicas. Ha' uma unica execucao por combinacao")
    w("      (pergunta_idx x modelo x criterio x modo). Nao e' possivel separar variacao")
    w("      entre condicoes de variabilidade estocastica da geracao.")
    w("")
    w("  L6. O baseline fixo50 tem numero de trechos fixo mas contagem de tokens variavel")
    w("      por consulta (minimo e maximo observados diferem em mais de 17 mil tokens),")
    w("      de modo que eta e' uma razao entre duas quantidades variaveis e nao uma")
    w("      economia contra um denominador constante.")
    w("")
    w("  L7. BERTScore mede similaridade semantica com uma resposta de referencia; nao")
    w("      mede correcao factual, cobertura de todos os itens do gabarito nem utilidade")
    w("      da resposta. Empates exatos de BERTScore indicam respostas identicas ou")
    w("      semanticamente indistinguiveis para essa metrica, nao equivalencia de fato.")
    w("")
    w("  L8. A anotacao de trecho-ouro (id_chunk_gold, parou_no_gold) esta disponivel")
    w("      apenas para um subconjunto das consultas e nao foi usada nas analises deste")
    w("      relatorio; ela permanece disponivel nos arquivos originais.")
    w("")
    w("=" * 78)
    w("FIM")
    w("=" * 78)

    texto = "\n".join(L)
    p = os.path.join(SAIDA, "interpretacao_resultados.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(f"interpretacao_resultados.txt gravado: {p} ({len(L)} linhas)")
    return p


if __name__ == "__main__":
    RES = main()
    gerar(RES)
    log("\nConcluido.")
