"""
05_piloto.py -- §G da qualificação: piloto obrigatório antes de liberar a
máquina para as ~55-110 h da rodada completa.

Por que existe: os itens 01, 06, 07 e 08 alteraram o laço de medição -- trocam
a fonte dos logits (`output_logits` em vez de `output_scores`) e mudam a
extensão e o conteúdo do que é lido por passo. São grátis em GPU, e é
exatamente por isso que merecem verificação: um deslocamento de um passo entre
a pilha de logits e a sequência gerada contamina a coluna principal da tese em
silêncio, e o `--smoke` não pega erro de índice de logit porque não há logit
nenhum ali.

O piloto roda 5 perguntas no gpt-oss:20b -- o menor download e, mais
importante, o ÚNICO modelo que exercita o alinhamento de canal do item 03, que
é a peça mais frágil do conjunto.

O que ele confere, em ordem:

  1. INVARIANTE DE ÍNDICE. Com decodificação gulosa, argmax(logit_i) tem que
     ser o token gerado no passo i. Contado em todos os passos. Divergência
     não é motivo para abortar -- é o sintoma de interferência de logits
     processor que o item 01 descreve -- mas precisa aparecer com número.
  2. ALINHAMENTO DE CANAL (item 03). Fração de posições com inicio > 0. Se
     vier zero num modelo que declara marcadores, a entropia mediu a abertura
     do canal de análise em vez da resposta.
  3. n_passos_entropia (item 04) contra o mínimo de agregação.
  4. SONDAS (itens 13 e 14): massa de probabilidade somada sobre as variantes
     de superfície. Massa baixa = número não interpretável.
  5. MODALIDADE B (seção 4.3.2): conformidade de formato e, sobretudo, a
     fração de posições em que a janela de entropia pôde ser deslocada para
     DEPOIS de {"resposta": ". Onde não pôde, a entropia mede sintaxe de JSON
     e não é comparável com a Modalidade A. Este é o item mais frágil da
     adição de 2026-09-05 e é o que decide se o Bloco 2 sob JSON tem valor
     para a comparação.
  6. IDENTIDADE DO RETRIEVER (item 22a): max_similaridade_e_gold tem que ser
     igual a "o gold está em 1º no ranking natural", em 100% das linhas, e
     idêntico nas duas modalidades (o ranking não depende do critério).
  7. TETO DE TOKENS do Bloco 1: quantas varreduras incrementais bateriam no
     limite. É o número que decide se `--limite-tokens` precisa mudar antes
     da rodada longa.
  8. VARIÂNCIA DO EIXO D. Se D_iter for constante, o Bloco 1 inteiro não tem
     o que medir: com D=1 os braços iterativo e incremental são o MESMO
     experimento (W_1 = c_1, Eq. 20). Não é bug, é o retriever acertando de
     primeira -- mas é decisão a tomar ANTES de gastar 55-110 h, não depois.

Uso:
  python 05_piloto.py                      # gpt-oss, 5 perguntas
  python 05_piloto.py --modelos gemma3:27b --limite 3
  python 05_piloto.py --pular-bloco1       # só a parte de logits
  python 05_piloto.py --criterios regex    # piloto só da Modalidade A
  python 05_piloto.py --sem-espalhar       # amostra de cabeça (não recomendado)

Os DOIS blocos usam por padrão uma amostra IGUALMENTE ESPAÇADA.

No Bloco 1 porque as 192 estão ordenadas por origem: as 74 primeiras são as
sintéticas, escritas a partir do próprio chunk, e o retriever acerta quase
sempre nelas. Pilotar a cabeça mede o pedaço fácil e esconde a conferência 8.

No Bloco 2 porque, mesmo sendo as 74 todas sintéticas, o POSTO DO GOLD varia
muito entre elas. Sob e5-small as 8 primeiras têm o gold em [1,1,2,3,1,1,1,2]
-- todas no top-3 --, e um piloto de cabeça devolve o estrato
`B_reposicionado` VAZIO, sugerindo que ele não existe. Nas 74 inteiras ele é
17,6%. Foi assim que o piloto de 06/09/2026 saiu enganoso.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import comum

OK, FALHA, AVISO = "[ok]    ", "[FALHA] ", "[aviso] "


def _rodar(cmd):
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 70)
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        print(f"{FALHA}comando saiu com código {r.returncode}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="gpt-oss:20b")
    ap.add_argument("--limite", type=int, default=5)
    ap.add_argument("--pular-bloco1", action="store_true")
    ap.add_argument("--criterios", default=",".join(comum.CRITERIOS),
                    help="modalidades de parada a pilotar no Bloco 2 (padrão: as duas)")
    ap.add_argument("--sem-espalhar", action="store_true",
                    help="usa as N PRIMEIRAS perguntas do Bloco 1 em vez de N "
                         "igualmente espaçadas. O padrão é espalhar, porque a "
                         "cabeça das 192 é o subconjunto sintético e fácil")
    ap.add_argument("--smoke", action="store_true",
                    help="exercita o próprio piloto sem GPU (não substitui o piloto real)")
    args = ap.parse_args()

    os.makedirs(comum.DIR_SAIDAS, exist_ok=True)
    tag = args.modelos.split(",")[0].strip()
    nome = comum.nome_saida(tag)
    extra = ["--smoke"] if args.smoke else []

    print("=" * 70)
    print(f"PILOTO -- {tag}, {args.limite} perguntas")
    print("=" * 70)

    # --espalhar: as 192 estão ordenadas por origem (74 sintéticas primeiro,
    # depois perguntas reais de usuário). Pilotar a cabeça mede o pedaço fácil
    # e dá D=1 em toda parte -- ver conferência 8.
    espalhar = [] if args.sem_espalhar else ["--espalhar"]
    if not args.pular_bloco1:
        _rodar(["02_bloco1_desempenho.py", "--modelos", tag,
                "--limite", str(args.limite), "--dmax", "10",
                "--sem-bertscore", "--forcar"] + espalhar + extra)
    # --espalhar TAMBÉM no Bloco 2 (corrigido em 06/09/2026). As 74 são todas
    # sintéticas, e por isso eu tinha julgado que espalhar não importava aqui.
    # Importa: o POSTO DO GOLD varia muito entre elas. As 8 primeiras têm o
    # gold em [1,1,2,3,1,1,1,2] -- todas no top-3 --, então o estrato
    # B_reposicionado sai VAZIO e o piloto sugere que ele não existe. Numa
    # amostra espalhada aparece ~12%, contra os 17,6% das 74 inteiras.
    _rodar(["03_bloco2_entropia.py", "--modelos", tag,
            "--limite", str(args.limite), "--criterios", args.criterios,
            "--sem-bertscore", "--forcar"] + espalhar + extra)

    print("\n" + "=" * 70)
    print("CONFERÊNCIAS DO §G")
    print("=" * 70)
    problemas = []      # defeito CORRIGÍVEL -> reprova o piloto
    achados = []        # propriedade dos dados -> registra, não reprova

    arq_ent = os.path.join(comum.DIR_SAIDAS, f"b2_entropia_{nome}.parquet")
    if not os.path.exists(arq_ent):
        print(f"{FALHA}{arq_ent} não existe -- o Bloco 2 não chegou a gravar.")
        sys.exit(1)
    ent = pd.read_parquet(arq_ent)
    res = pd.read_parquet(os.path.join(comum.DIR_SAIDAS, f"b2_resultados_{nome}.parquet"))
    if "criterio" not in ent.columns:            # saídas anteriores a 2026-09-05
        ent["criterio"] = "regex"
    if "criterio" not in res.columns:
        res["criterio"] = "regex"
    criterios = sorted(ent["criterio"].dropna().unique().tolist())
    print(f"\nModalidades de parada no arquivo: {criterios}")
    # as sondas dos itens 13/14 são geradas UMA vez por posição e repetidas nas
    # linhas das duas modalidades: contá-las duas vezes não muda a média, mas
    # infla o n e daria uma falsa sensação de amostra
    pos = ent.drop_duplicates(["pergunta_idx", "posicao"])

    # ---- 1. invariante de índice ------------------------------------
    div = int(ent["n_divergencias_argmax"].sum())
    ver = int(ent["n_passos_verificados"].sum())
    taxa = 100 * div / ver if ver else float("nan")
    print(f"\n1. Invariante argmax(logit_i) == token_i (greedy)")
    print(f"   {div} divergências em {ver} passos ({taxa:.3f}%)")
    if len(criterios) > 1:
        for c in criterios:
            s = ent[ent["criterio"] == c]
            v = int(s["n_passos_verificados"].sum())
            print(f"     [{c:<5}] {int(s['n_divergencias_argmax'].sum())}/{v}")
    if div == 0:
        print(f"{OK}logits crus limpos: nenhum logits processor interferiu.")
    else:
        print(f"{AVISO}há interferência de processor (repetition_penalty / "
              f"min_new_tokens / no_repeat_ngram).")
        print("        Não invalida a rodada -- `output_logits` justamente contorna o")
        print("        efeito na MEDIDA -- mas registre a taxa junto do resultado e")
        print("        confira o generation_config no metadata_b2_*.json.")

    # ---- 2. alinhamento de canal (item 03) --------------------------
    reg = ent[ent["criterio"] == "regex"] if "regex" in criterios else ent
    frac = float((reg["inicio_janela_entropia"] > 0).mean()) if len(reg) else 0.0
    print(f"\n2. Alinhamento de canal (item 03)")
    print(f"   posições com inicio > 0 (Modalidade A, só o canal): {100 * frac:.1f}%")
    if "json" in criterios:
        fj = float((ent.loc[ent["criterio"] == "json", "inicio_janela_entropia"] > 0).mean())
        print(f"   idem na Modalidade B (canal + abertura da chave): {100 * fj:.1f}%")
        print("   -> na B o deslocamento é o caso normal, e por isso ele é conferido")
        print("      pela conferência 5, que olha janela_json_alinhada e não inicio.")
    meta_path = os.path.join(comum.DIR_SAIDAS, f"metadata_b2_{nome}.json")
    declara = False
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        declara = bool(meta.get("marcadores_fim_prefacio"))
        print(f"   marcadores declarados: {meta.get('marcadores_fim_prefacio')} -> "
              f"ids {meta.get('ids_marcadores_resolvidos')}")
    if declara and frac == 0:
        problemas.append("alinhamento de canal não aconteceu (inicio > 0 em 0% das posições)")
        print(f"{FALHA}o modelo declara marcadores mas NENHUMA posição pulou o prefácio.")
        print("        A entropia mediu a abertura do canal de análise. NÃO rode a")
        print("        rodada longa assim.")
    elif declara:
        print(f"{OK}alinhamento ativo.")
    else:
        print(f"{OK}modelo não declara marcadores; inicio = 0 é o esperado.")

    # ---- 3. n_passos_entropia (item 04) -----------------------------
    print(f"\n3. n_passos_entropia (mínimo para agregar = {comum.MIN_PASSOS_ENTROPIA})")
    print(ent.groupby("criterio")["n_passos_entropia"]
          .describe().to_string())
    curtos = int((ent["n_passos_entropia"] < comum.MIN_PASSOS_ENTROPIA).sum())
    print(f"   posições abaixo do mínimo: {curtos}/{len(ent)}")
    if curtos > 0.3 * len(ent):
        print(f"{AVISO}mais de 30% das posições ficariam fora da média. Reveja")
        print("        MIN_PASSOS_ENTROPIA ou o alinhamento.")

    # ---- 4. sondas (itens 13 e 14) ----------------------------------
    print(f"\n4. Sondas de um forward  (n = {len(pos)} posições, sem duplicar "
          f"pelas modalidades)")
    if "sonda_massa_total" in pos.columns:
        mt = pos["sonda_massa_total"]
        print(f"   massa sim+não (item 14): média {mt.mean():.3f}  "
              f"mín {mt.min():.3f}  abaixo de 0,1: {int((mt < 0.1).sum())}/{len(mt)}")
        if mt.mean() < 0.3:
            problemas.append("massa da sonda sim/não muito baixa -- número não interpretável")
            print(f"{FALHA}o modelo não está respondendo no formato pedido. P(sim) não é")
            print("        interpretável assim -- ajuste o prompt ou o prefixo de canal.")
        else:
            print(f"{OK}massa suficiente.")
    if "logprob_medio_referencia" in pos.columns:
        lp = pos["logprob_medio_referencia"]
        print(f"   logprob médio/token do gabarito (item 13): "
              f"gold {pos.loc[pos.is_gold, 'logprob_medio_referencia'].mean():.3f}  "
              f"não-gold {pos.loc[~pos.is_gold, 'logprob_medio_referencia'].mean():.3f}")
        if lp.isna().all():
            problemas.append("teacher forcing devolveu só nulos")
            print(f"{FALHA}nenhum valor calculado.")
        else:
            print(f"{OK}teacher forcing produzindo valores.")
    if "sobreposicao_lexical" in pos.columns:
        print(f"   sobreposição lexical (controle): "
              f"gold {pos.loc[pos.is_gold, 'sobreposicao_lexical'].mean():.3f}  "
              f"não-gold {pos.loc[~pos.is_gold, 'sobreposicao_lexical'].mean():.3f}")
        print("   -> se a diferença aqui já for grande, boa parte do item 13 é cópia")
        print("      lexical do gabarito, não ancoragem. Reportar como covariável.")

    # ---- 5. Modalidade B: formato e janela de entropia --------------
    print("\n5. Modalidade B -- saída estruturada em JSON (seção 4.3.2)")
    if "json" not in criterios:
        print(f"{AVISO}esta rodada do piloto não incluiu a Modalidade B.")
        print("        A rodada longa roda as duas por padrão -- pilote as duas antes")
        print("        de liberar a máquina, ou o único teste do caminho JSON terá")
        print("        sido o --smoke, que não gera token nenhum.")
    else:
        js = ent[ent["criterio"] == "json"]
        conf = 100 * (js["conformidade_json"] == True).mean()      # noqa: E712
        estr = 100 * (js["json_estrito"] == True).mean()           # noqa: E712
        jan = 100 * (js["janela_json_alinhada"] == True).mean()    # noqa: E712
        print(f"   conformidade de formato: {conf:.1f}%   (JSON estrito {estr:.1f}%)")
        print(f"   janela de entropia alinhada ao valor da chave: {jan:.1f}%")
        if conf < 50:
            problemas.append(f"conformidade JSON de apenas {conf:.0f}% -- a Modalidade "
                             f"B não está sendo exercida")
            print(f"{FALHA}o modelo quase não emite o formato pedido. A Eq. 22 estaria")
            print("        decidindo sobre texto que não é JSON: o D da Modalidade B")
            print("        não mede o que a tese diz que mede. Ajuste o prompt antes.")
        elif jan < 80:
            problemas.append(f"janela de entropia alinhada em só {jan:.0f}% das posições "
                             f"da Modalidade B")
            print(f"{FALHA}nas posições não alinhadas a entropia mede a sintaxe do JSON,")
            print("        não a resposta, e NÃO é comparável com a Modalidade A. Com")
            print("        menos de 80% o filtro do 04_metricas.py deixa um subconjunto")
            print("        selecionado pelo próprio modelo, o que enviesa a comparação.")
        else:
            print(f"{OK}formato e janela consistentes: as duas modalidades medem a")
            print("        entropia dos mesmos 20 primeiros tokens DE RESPOSTA.")
        # D das duas modalidades sobre as mesmas perguntas
        piv = res.pivot_table(index="pergunta_idx", columns="criterio",
                              values="D_embaralhado")
        if {"regex", "json"} <= set(piv.columns):
            amb = piv.dropna()
            print(f"   D_embaralhado: as duas pararam em {len(amb)}/{len(piv)} perguntas"
                  + (f", iguais em {100 * (amb['regex'] == amb['json']).mean():.0f}%"
                     if len(amb) else ""))

    # ---- 5b. circularidade: a entropia mede relevância ou parada? ----
    print("\n5b. A entropia mede relevância, ou mede 'o modelo recusou'?")
    print(f"   AUC de 'recusa tem entropia menor'. 0,5 = sem informação; alerta "
          f"fora de [{1 - comum.AUC_CIRCULARIDADE_ALERTA:.2f}; "
          f"{comum.AUC_CIRCULARIDADE_ALERTA:.2f}], nos DOIS sentidos.")
    for c in criterios:
        s = ent[ent["criterio"] == c]
        d = comum.diagnostico_circularidade(s["entropia_media"], s["valida"])
        if d["auc"] is None:
            print(f"   [{c:<5}] só um dos dois grupos existe -- sem AUC "
                  f"({d['n_validas']} válidas, {d['n_recusas']} recusas)")
            continue
        print(f"   [{c:<5}] AUC={d['auc']:.3f}  teto recusa={d['teto_recusa']:.6f}  "
              f"piso válida={d['piso_valida']:.6f}"
              f"{'  SEPARAÇÃO TOTAL' if d['separacao_total'] else ''}")
        avisos = comum.texto_alerta_circularidade(d, c)
        if avisos:
            for linha in avisos:
                print("   " + linha.strip())
            # NÃO entra em `problemas`: ver a nota logo abaixo.
            achados.append(f"entropia circular na modalidade {c} "
                           f"(AUC={d['auc']:.3f}"
                           f"{', separação total' if d['separacao_total'] else ''})")
    if not achados:
        print(f"{OK}nenhuma modalidade circular.")
    print("\n   POR QUE ISTO NÃO REPROVA O PILOTO (mudou em 05/09/2026).")
    print("   Este portão existe para barrar defeito CORRIGÍVEL antes de gastar")
    print("   55-110 h de GPU. Circularidade não é defeito corrigível: sob a")
    print("   Modalidade B ela vem de a sentinela ser string fixa, que É a Eq. 22")
    print("   -- não há prompt nem config que a remova sem abandonar a definição.")
    print("   Bloquear aqui mandaria consertar o que não tem conserto, e o único")
    print("   jeito de seguir seria ignorar o portão, o que ensina a ignorar")
    print("   portões. Além disso a nula estratificada do item 21b já neutraliza")
    print("   a estatística degenerada sozinha (ela devolve p=1,000, isto é")
    print("   'não acrescenta nada', em vez de 'pior que o acaso'). O portão")
    print("   seria redundante com a inferência correta.")
    print("   Um AUC intermediário -- 0,7, 0,8 -- muito menos reprova: é")
    print("   acoplamento esperado. Não leia este AUC como conclusão sobre sinal.")

    # ---- 6. identidade do retriever (item 22a) ----------------------
    print("\n6. Identidade max_similaridade_e_gold == (gold em 1º) -- item 22a")
    uma = res.drop_duplicates("pergunta_idx")
    igual = (uma["max_similaridade_e_gold"].astype(bool)
             == (uma["posicao_gold_no_ranking_natural"] == 1))
    print(f"   confere em {int(igual.sum())}/{len(uma)} perguntas")
    if not igual.all():
        problemas.append("a identidade do item 22a não fecha")
        print(f"{FALHA}algo mudou no ranking ou na montagem do top-10.")
    else:
        print(f"{OK}como esperado -- o 'baseline de similaridade' é o Precision@1.")
    if len(criterios) > 1:
        inv = res.groupby("pergunta_idx")["max_similaridade_e_gold"].nunique()
        n_div = int((inv > 1).sum())
        print(f"   invariância ao critério: {n_div} divergência(s) em {len(inv)}")
        if n_div:
            problemas.append("o ranking mudou entre as modalidades -- não deveria")
            print(f"{FALHA}o top-10 e o sorteio têm que ser idênticos nas duas")
            print("        modalidades: só a geração muda.")

    # ---- 7. teto de tokens do Bloco 1 -------------------------------
    arq_b1 = os.path.join(comum.DIR_SAIDAS, f"b1_resumo_{nome}.parquet")
    if os.path.exists(arq_b1):
        b1 = pd.read_parquet(arq_b1)
        inc = b1[b1["modo"] == "incremental"]
        if len(inc):
            t = 100 * inc["censurado_limite_tokens"].mean()
            print(f"\n7. Teto de tokens no braço incremental: {t:.1f}% censuradas")
            if t > 5:
                print(f"{AVISO}o braço incremental está sendo cortado por VRAM, não por")
                print("        ancoragem. Considere --limite-tokens menor, ou aceite e")
                print("        reporte a taxa. Lembre que o piloto usou --dmax 10; na")
                print("        rodada real (--dmax 50) esta taxa tende a subir.")
            else:
                print(f"{OK}dentro do aceitável.")

    # ---- 8. o eixo D tem variância? ---------------------------------
    if os.path.exists(arq_b1):
        b1 = pd.read_parquet(arq_b1)
        it = b1[b1["modo"].isin(["iterativo", "incremental"])]
        d = it["D_iter"].dropna()
        print("\n8. O eixo D varia? (o Bloco 1 inteiro depende disso)")
        if len(d):
            print(f"   D_iter observado: {sorted(d.unique().tolist())}  "
                  f"n={len(d)}  constante={d.nunique() == 1}")
            if d.nunique() == 1:
                achados.append(f"D_iter constante (= {d.iloc[0]:.0f}) em todas as "
                               f"{len(d)} varreduras do piloto")
                print(f"{AVISO}D não variou nesta amostra.")
                print("        Se isso valer para as 192, a comparação iterativo x")
                print("        incremental do Bloco 1 não tem variância: com D=1 os dois")
                print("        braços são o MESMO experimento (W_1 = c_1 pela Eq. 20), e")
                print("        o eta vira uma razão fixa. Não é bug -- é o retriever")
                print("        acertando de primeira -- mas decida ANTES das 55-110 h se")
                print("        um resultado nulo nesse eixo serve à tese.")
                print("        Barato de checar: rode o Bloco 1 com --espalhar --limite 20")
                print("        --criterios regex --dmax 10, que cobre as perguntas reais")
                print("        de usuário (curtas e vagas), onde D tem chance de variar.")
            else:
                print(f"{OK}D varia -- a comparação do Bloco 1 tem o que medir.")
        else:
            print(f"{AVISO}nenhuma varredura parou; D é todo censurado nesta amostra.")

    print("\n" + "=" * 70)
    if achados:
        print("ACHADOS -- propriedades dos dados, NÃO defeitos. Não bloqueiam a")
        print("rodada; entram no texto da tese e devem ser lidos com o item 21b:")
        for a in achados:
            print(f"  - {a}")
        print()
    if problemas:
        print(f"{len(problemas)} problema(s) CORRIGÍVEL(is) a resolver ANTES da "
              f"rodada longa:")
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)
    print("Piloto OK. Pode liberar a máquina para a rodada completa.")
    print("Mande também os metadata_*.json do piloto para o Murilo.")


if __name__ == "__main__":
    main()
