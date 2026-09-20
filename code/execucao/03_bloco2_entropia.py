"""
03_bloco2_entropia.py -- BLOCO 2: Sonda de Entropia (Logits) e Análise de
Incerteza.

Objetivo: isolar o estado de confiança do modelo frente à POSIÇÃO do contexto.
A lógica espelha estritamente o experimento.py fornecido pelo Murilo; o que
mudou é apenas a infraestrutura (multi-modelo, gestão de VRAM, leitura dos
Parquet padronizados). A mecânica experimental é idêntica.

Dataset : perguntas_74_gold.parquet (74 questões sintéticas com chunk-ouro)
Texto   : coluna `texto_artigo` do corpus (chunk restrito, ~100 tokens)

Mecânica (Top-10 Shuffle):
  1. ranking completo por similaridade de cosseno sobre o corpus todo
     (não top-100: precisamos da posição real do gold mesmo quando ele cai
     em 300º lugar);
  2. top-10 natural;
  3. se nenhum chunk-ouro estiver no top-10, o 10º colocado (o mais fraco) é
     SUBSTITUÍDO pelo chunk-ouro principal;
  4. os 10 candidatos são embaralhados com semente determinística
     SEED_BASE(42) + índice da pergunta;
  5. GERAÇÃO EXAUSTIVA: as 10 posições são geradas, sem early-stopping --
     precisamos da entropia das 10 para achar o mínimo;
  6. SIMULAÇÃO RETROATIVA: o regex de parada é reaplicado sobre as 10
     respostas já geradas para calcular onde o método teria parado
     (D_embaralhado). Isso não custa GPU nenhuma;
  7. compara D_embaralhado com a posição de menor entropia média.

Entropia: Shannon EXATA (softmax sobre o vocabulário inteiro, não sobre um
top-k) nos primeiros N_TOKENS_ENTROPIA = 20 passos de geração. A lista
completa por token é salva, o que permite recalcular a média sobre prefixos
de 5/10/20 tokens na análise sem tocar na GPU.

Por que aqui output_scores=True é seguro: o contexto é UM chunk (~100
tokens) e cada tensor de scores é reduzido a um float e descartado no mesmo
passo do laço. É exatamente o oposto do Bloco 1.

Leitura das métricas (importante, e igual ao docstring do experimento.py):
  * PRIMÁRIAS, em nível de chunk e invariantes ao sorteio:
      min_entropia_e_gold      -- o chunk de menor entropia é o chunk-ouro?
      max_similaridade_e_gold  -- BASELINE do retriever. Se a entropia não
                                  superar este número, o sinal de entropia
                                  não acrescenta nada ao que a similaridade
                                  de cosseno já entrega.
  * SECUNDÁRIA, em nível de posição e sensível ao sorteio:
      concordancia_D_entropia  -- D_embaralhado == posicao_min_entropia.
                                  Boa parte dessa taxa é ruído da permutação:
                                  se várias posições produzem resposta
                                  válida, D cai na que o sorteio colocou
                                  mais cedo.

Saídas (em saidas/):
  b2_resultados_<modelo>.parquet  1 linha por pergunta
  b2_entropia_<modelo>.parquet    1 linha por posição embaralhada (10 por
                                  pergunta, formato longo para os gráficos)

Uso:
  python 03_bloco2_entropia.py
  python 03_bloco2_entropia.py --modelos gemma3:27b --limite 3 --smoke
"""

import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd

import comum

RESPOSTA_FALLBACK = "A informação não foi encontrada nos documentos disponíveis."


# ============================================================
# Monta o conjunto de 10 candidatos e embaralha
# ============================================================

def montar_top10_com_gold(ordem, sims, ids_corpus, ids_gold, id_gold_principal, rng):
    """
    Retorna uma lista de dicts (um por candidato, já EMBARALHADA) e metadados.

    `_posicao_natural` guarda a posição do candidato no ranking ANTES do
    embaralhamento -- para os 10 naturais é 1..10; para um gold inserido por
    substituição é a posição REAL dele no ranking completo (ex.: 37 de 740),
    que é a informação de interesse.
    """
    k = comum.K_TOP
    topo = ordem[:k]
    candidatos = [
        {
            "indice_corpus": int(pos),
            "id_chunk": int(ids_corpus[pos]),
            "similaridade": float(sims[j]),
            "_posicao_natural": j + 1,
        }
        for j, pos in enumerate(topo)
    ]

    gold_ja_no_top10 = any(c["id_chunk"] in ids_gold for c in candidatos)
    id_removido = None

    if gold_ja_no_top10:
        foi_reposicionado = False
    else:
        # posição real do gold principal no ranking completo
        alvo = np.where(ids_corpus[ordem] == int(id_gold_principal))[0]
        if len(alvo) == 0:
            raise ValueError(
                f"chunk-ouro {id_gold_principal} não existe no corpus -- "
                f"rode 00_preparar_dados.py de novo e confira a validação."
            )
        j = int(alvo[0])
        linha_gold = {
            "indice_corpus": int(ordem[j]),
            "id_chunk": int(id_gold_principal),
            "similaridade": float(sims[j]),
            "_posicao_natural": j + 1,
        }
        id_removido = candidatos[k - 1]["id_chunk"]
        candidatos = candidatos[: k - 1] + [linha_gold]
        foi_reposicionado = True

    for c in candidatos:
        c["is_gold"] = c["id_chunk"] in ids_gold

    perm = rng.permutation(len(candidatos))
    candidatos = [candidatos[i] for i in perm]
    return candidatos, foi_reposicionado, id_removido


# ============================================================
# Geração exaustiva das 10 posições + simulação retroativa
# ============================================================

def rodar_top10(gerador, pergunta, candidatos, textos_artigo, referencia="",
                com_sondas=True, prefixo_forcado=None, criterios=None):
    """
    Gera as 10 posições (sem early-stopping) sob CADA critério de parada, e
    roda as duas sondas de um forward por posição (itens 13 e 14).

    DUAS MODALIDADES (a mesma distinção do Bloco 1, seção 4.3.2 da tese):
      "regex" -> autômato sobre a resposta em prosa (Eq. 21)
      "json"  -> chave "resposta" == "NÃO ENCONTRADO" (Eq. 22)

    Cada modalidade exige a sua própria geração, porque o prompt é outro: são
    10 x len(criterios) gerações por pergunta. Com as duas, 20.

    O QUE **NÃO** É DUPLICADO: as sondas dos itens 13 e 14 e a sobreposição
    lexical não dependem do critério de parada -- o teacher forcing e o
    P(sim) são um forward sobre (pergunta, chunk), sem geração. Rodam uma vez
    e o valor é repetido nas linhas das duas modalidades. Duplicá-las seria
    gastar GPU para obter o mesmo número duas vezes.

    Saída em formato LONGO: uma linha por (posição, critério), com a coluna
    `criterio` -- mesma convenção do Bloco 1, o que deixa o 04_metricas.py
    agrupar os dois blocos do mesmo jeito.
    """
    criterios = criterios or list(comum.CRITERIOS)
    registros = []
    eta_por_criterio = {c: 0 for c in criterios}

    for posicao, cand in enumerate(candidatos, start=1):
        texto = str(textos_artigo[cand["indice_corpus"]])

        # --- comum às duas modalidades: roda UMA vez ------------------
        base = {
            "posicao": posicao,
            "id_chunk": cand["id_chunk"],
            "is_gold": bool(cand["is_gold"]),
            "posicao_natural": cand["_posicao_natural"],
            "similaridade": cand["similaridade"],
            "sobreposicao_lexical": comum.sobreposicao_lexical(referencia, texto),
        }
        if com_sondas:
            base.update(gerador.logprob_referencia(pergunta, texto, referencia))
            base.update(gerador.sonda_sim_nao(pergunta, texto))

        # --- uma geração por modalidade -------------------------------
        for criterio in criterios:
            r = gerador.gerar_com_entropia(pergunta, texto, criterio=criterio,
                                           prefixo_forcado=prefixo_forcado)
            v = comum.validar_resposta(r["texto_resposta"], criterio)
            eta_por_criterio[criterio] += r["n_tokens_prompt"] + r["n_tokens_resposta"]

            registros.append({
                **base,
                "criterio": criterio,
                # --- janela primária (20 passos), definida a priori ---
                "entropia_media": r["entropia_media"],
                "logprob_medio": r["logprob_medio"],
                "p_top1_medio": r["p_top1_medio"],
                "margem_media": r["margem_media"],
                "n_passos_entropia": r["n_passos_entropia"],
                # --- registro completo (secundária pré-declarada, item 06) ---
                "entropia_por_token": r["entropia_por_token"],
                "logprob_por_token": r["logprob_por_token"],
                "p_top1_por_token": r["p_top1_por_token"],
                "margem_por_token": r["margem_por_token"],
                "n_passos_registrados": r["n_passos_registrados"],
                "entropia_media_todos": r["entropia_media_todos"],
                # --- item 08 ---
                "top5_tokens_primeiro_passo": r["top5_tokens_primeiro_passo"],
                "top5_probs_primeiro_passo": r["top5_probs_primeiro_passo"],
                # --- itens 03, 09, 11 e §G ---
                "inicio_janela_entropia": r["inicio_janela_entropia"],
                # sob JSON: a janela foi deslocada para o valor da chave?
                # False = o modelo não emitiu o formato e a entropia daquela
                # posição mede sintaxe, não resposta. Filtrar na análise.
                "janela_json_alinhada": r["janela_json_alinhada"],
                "prefixo_resposta": r["prefixo_resposta"],
                "atingiu_max_tokens": r["atingiu_max_tokens"],
                "n_divergencias_argmax": r["n_divergencias_argmax"],
                "n_passos_verificados": r["n_passos_verificados"],
                # --- decisão de parada, por modalidade ---
                "valida": v["valida"],
                "conformidade_json": v["conformidade_json"],
                "json_estrito": v["json_estrito"],
                "recusa_textual_no_valor": v["recusa_textual_no_valor"],
                "motivo_decisao": v["motivo_decisao"],
                "texto_resposta": v["resposta_extraida"],
                "texto_resposta_bruto": r["texto_resposta"],
                "n_tokens_prompt": r["n_tokens_prompt"],
                "n_tokens_resposta": r["n_tokens_resposta"],
                "n_tokens_posicao": r["n_tokens_prompt"] + r["n_tokens_resposta"],
                "eta_tokens_acumulado": eta_por_criterio[criterio],
            })

    # ---- SIMULAÇÃO RETROATIVA, uma por modalidade ----------------------
    # Nada é gerado aqui: só reaplicamos a regra de parada sobre as respostas
    # que já existem. Como cada prompt contém UM chunk, a resposta da posição
    # i não depende das anteriores -- então isto reproduz exatamente o que o
    # laço iterativo com early-stopping teria produzido naquela ordem.
    saidas = {}
    for criterio in criterios:
        regs = [r for r in registros if r["criterio"] == criterio]

        D_embaralhado = None
        id_chunk_D = None
        resposta_D = RESPOSTA_FALLBACK
        eta_tokens_ate_D = eta_por_criterio[criterio]     # default: esgotou as 10
        for reg in regs:
            if reg["valida"]:
                D_embaralhado = reg["posicao"]
                id_chunk_D = reg["id_chunk"]
                resposta_D = reg["texto_resposta"]
                eta_tokens_ate_D = reg["eta_tokens_acumulado"]
                break

        # D sob a regra alternativa do JSON (sentinela OU recusa em prosa
        # dentro do valor). Custo zero, mesma lógica do Bloco 1.
        D_alt = None
        for reg in regs:
            if reg["valida"] and not reg["recusa_textual_no_valor"]:
                D_alt = reg["posicao"]
                break

        com_entropia = [r for r in regs if r["entropia_media"] is not None]
        reg_min = min(com_entropia, key=lambda r: r["entropia_media"]) if com_entropia else None
        com_sim = [r for r in regs if r["similaridade"] is not None]
        reg_max_sim = max(com_sim, key=lambda r: r["similaridade"]) if com_sim else None

        n_infer = len(regs)
        saidas[criterio] = {
            "D_embaralhado": D_embaralhado,
            "D_embaralhado_regra_alternativa": D_alt,
            "id_chunk_D": id_chunk_D,
            "texto_resposta_final": resposta_D,
            "eta_tokens_ate_D": eta_tokens_ate_D,
            "eta_tokens_total_10": eta_por_criterio[criterio],
            "posicao_min_entropia": reg_min["posicao"] if reg_min else None,
            "entropia_minima": reg_min["entropia_media"] if reg_min else None,
            "min_entropia_e_gold": bool(reg_min["is_gold"]) if reg_min else None,
            "max_similaridade_e_gold": bool(reg_max_sim["is_gold"]) if reg_max_sim else None,
            "n_posicoes_validas": sum(1 for r in regs if r["valida"]),
            # conformidade de formato como métrica autônoma (seção 4.3.2)
            "n_pos_conformes": sum(1 for r in regs if r["conformidade_json"] is True),
            "n_pos_json_estrito": sum(1 for r in regs if r["json_estrito"] is True),
            "n_pos_formato_invalido": sum(1 for r in regs if r["conformidade_json"] is False),
            # sob JSON: em quantas posições a janela de entropia pôde ser
            # deslocada para o valor da chave? Se for baixo, a entropia
            # daquele modelo mede sintaxe e não é comparável à Modalidade A.
            "n_pos_janela_json_alinhada": sum(
                1 for r in regs if r["janela_json_alinhada"] is True),
            "n_inferencias": n_infer,
        }

    return {"registros_posicao": registros, "por_criterio": saidas}


# ============================================================
# Uma rodada completa: um modelo x 74 perguntas x 10 posições
# ============================================================

def rodar_modelo(tag, df_perguntas, df_emb, corpus, matriz,
                 smoke=False, com_bertscore=True, com_sondas=True,
                 prefixo_forcado=None, quant=None, attn=None, recomecar=False,
                 criterios=None):
    criterios = criterios or list(comum.CRITERIOS)
    nome = comum.nome_saida(tag)
    # A condição de prefixo forçado (item 15) grava em arquivos próprios: é uma
    # CONDIÇÃO EXPERIMENTAL distinta, não uma variante da mesma medida.
    if prefixo_forcado:
        nome = f"{nome}__prefixo"
    arq_res = os.path.join(comum.DIR_SAIDAS, f"b2_resultados_{nome}.parquet")
    arq_ent = os.path.join(comum.DIR_SAIDAS, f"b2_entropia_{nome}.parquet")

    if smoke:
        from _gerador_falso import GeradorFalso
        gerador = GeradorFalso(tag)
    else:
        gerador = comum.Gerador(tag, quant=quant, attn=attn)

    ids_corpus = corpus["id"].to_numpy()
    textos_artigo = corpus["texto_artigo"].astype(str).to_numpy()
    emb_por_idx = dict(zip(df_emb["pergunta_idx"], df_emb["embedding_pergunta"]))

    # ---- retomada por pergunta (1 linha por pergunta no resultado) ----
    if recomecar:
        resultados, entropia_long, concluidas = [], [], set()
        comum.limpar_parciais(arq_res, arq_ent)
        print("  [checkpoint] --recomecar: parciais apagados, começando do zero.")
    else:
        resultados, entropia_long, concluidas = comum.retomar_checkpoint(
            arq_res, arq_ent, linhas_por_pergunta=len(criterios))
    n_retomadas = len(concluidas)

    t_inicio = time.time()
    desde_checkpoint = 0

    def _gravar_parcial(qtd):
        ok = comum.salvar_parquet_atomico(
            pd.DataFrame(resultados), comum.caminho_parcial(arq_res))
        comum.salvar_parquet_atomico(
            pd.DataFrame(entropia_long), comum.caminho_parcial(arq_ent))
        if ok:
            print(f"  (checkpoint: {qtd} perguntas concluídas neste modelo)")

    try:
        for _, row in df_perguntas.iterrows():
            idx = int(row["pergunta_idx"])
            if idx in concluidas:
                continue
            pergunta = str(row["pergunta"])
            print(f"\n[{tag}] [{idx + 1}/{len(df_perguntas)}] {pergunta[:70]}...")

            t0 = time.time()
            comum.zerar_pico_vram()
            # semente determinística por pergunta: 42 + índice
            rng = np.random.RandomState(comum.SEED_BASE + idx)

            ordem, sims = comum.ranquear(emb_por_idx[idx], matriz, k=None)
            ids_gold = {int(x) for x in row["ids_chunk_gold_todos"]}

            # posição do melhor gold no ranking natural completo
            mascara = np.isin(ids_corpus[ordem], list(ids_gold))
            pos_golds = np.where(mascara)[0]
            posicao_gold_natural = int(pos_golds[0]) + 1 if len(pos_golds) else None
            posicoes_todos_golds = [int(p) + 1 for p in pos_golds]

            candidatos, foi_reposicionado, id_removido = montar_top10_com_gold(
                ordem, sims, ids_corpus, ids_gold, int(row["id_chunk_gold"]), rng
            )

            idx_golds_top10 = [i for i, c in enumerate(candidatos) if c["is_gold"]]
            saida = rodar_top10(
                gerador, pergunta, candidatos, textos_artigo,
                referencia=str(row["resposta_gabarito"]),
                com_sondas=com_sondas, prefixo_forcado=prefixo_forcado,
                criterios=criterios)
            dt = time.time() - t0

            for reg in saida["registros_posicao"]:
                entropia_long.append({"modelo": tag, "retriever": comum.RETRIEVER,
                                      "pergunta_idx": idx, **reg})

            # UMA LINHA POR (pergunta, critério) -- mesma convenção do Bloco 1
            for criterio in criterios:
                sc = saida["por_criterio"][criterio]
                concordancia = (
                    sc["D_embaralhado"] is not None
                    and sc["D_embaralhado"] == sc["posicao_min_entropia"]
                )
                resultados.append({
                    "modelo": tag,
                    "retriever": comum.RETRIEVER,
                    "criterio": criterio,
                    "pergunta_idx": idx,
                    "pergunta": pergunta,
                    "resposta_gabarito": str(row["resposta_gabarito"]),
                    "id_chunk_gold": int(row["id_chunk_gold"]),
                    "ids_chunk_gold_todos": [int(x) for x in row["ids_chunk_gold_todos"]],
                    "posicao_gold_no_ranking_natural": posicao_gold_natural,
                    "posicoes_todos_golds_natural": posicoes_todos_golds,
                    "gold_foi_reposicionado_no_top10": foi_reposicionado,
                    "id_chunk_removido_na_substituicao": id_removido,
                    "posicao_gold_no_top10_final": (
                        idx_golds_top10[0] + 1 if idx_golds_top10 else None),
                    "posicoes_golds_no_top10": [i + 1 for i in idx_golds_top10],
                    "n_golds_no_top10": len(idx_golds_top10),
                    "D_embaralhado": sc["D_embaralhado"],
                    "D_embaralhado_regra_alternativa": sc["D_embaralhado_regra_alternativa"],
                    "id_chunk_D": sc["id_chunk_D"],
                    "acertou_chunk_gold": (
                        sc["id_chunk_D"] is not None and int(sc["id_chunk_D"]) in ids_gold),
                    "n_posicoes_validas": sc["n_posicoes_validas"],
                    "posicao_min_entropia": sc["posicao_min_entropia"],
                    "entropia_minima": sc["entropia_minima"],
                    "min_entropia_e_gold": sc["min_entropia_e_gold"],
                    "max_similaridade_e_gold": sc["max_similaridade_e_gold"],
                    "concordancia_D_entropia": concordancia,
                    "n_pos_conformes": sc["n_pos_conformes"],
                    "n_pos_json_estrito": sc["n_pos_json_estrito"],
                    "n_pos_formato_invalido": sc["n_pos_formato_invalido"],
                    "n_pos_janela_json_alinhada": sc["n_pos_janela_json_alinhada"],
                    "n_inferencias": sc["n_inferencias"],
                    "texto_resposta_final": sc["texto_resposta_final"],
                    "eta_tokens_ate_D": sc["eta_tokens_ate_D"],
                    "eta_tokens_total_10": sc["eta_tokens_total_10"],
                    "tempo_segundos": dt,
                    "vram_pico_gb": comum.vram_pico_gb(),
                })

                extra = ""
                if criterio == "json":
                    extra = (f" conf={sc['n_pos_conformes']}/{sc['n_inferencias']}"
                             f" jan={sc['n_pos_janela_json_alinhada']}/{sc['n_inferencias']}")
                print(f"  {criterio:<5} D_embaralhado={sc['D_embaralhado']}  "
                      f"pos_min_entropia={sc['posicao_min_entropia']}  "
                      f"concorda={concordancia}  "
                      f"min_ent_e_gold={sc['min_entropia_e_gold']}  "
                      f"V={sc['n_posicoes_validas']}{extra}")
            print(f"  reposicionado={foi_reposicionado}  {dt:.1f}s  "
                  f"vram={comum.vram_pico_gb():.1f}GB")

            concluidas.add(idx)
            desde_checkpoint += 1
            if desde_checkpoint >= comum.CHECKPOINT_BLOCO2:
                _gravar_parcial(len(concluidas))
                desde_checkpoint = 0

    except (Exception, KeyboardInterrupt) as e:
        print(f"\n!!! INTERROMPIDO ({type(e).__name__}). Salvando o que já foi gerado.")
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
    finally:
        # grava o parcial SEMPRE ao sair do laço, Ctrl-C incluído
        try:
            if resultados:
                _gravar_parcial(len(concluidas))
        except Exception:
            traceback.print_exc()
        try:
            gerador.descarregar()
        except Exception:
            pass

    df_res = pd.DataFrame(resultados)
    df_ent = pd.DataFrame(entropia_long)

    # Checkpoint ANTES do BERTScore: são 740 gerações (10 x 74) sem
    # early-stopping. Perder isso por uma falha no cálculo da métrica final
    # seria caro demais.
    df_res.to_parquet(arq_res.replace(".parquet", "_checkpoint_sem_bertscore.parquet"),
                      index=False)
    comum.salvar_parquet_atomico(df_ent, arq_ent, permitir_encolher=True)
    print(f"\nCheckpoint sem BERTScore salvo. Entropia por posição: {arq_ent}")

    if com_bertscore and len(df_res):
        print("Calculando BERTScore (S)...")
        try:
            P, R, F1 = comum.calcular_bertscore(
                df_res["texto_resposta_final"].tolist(),
                df_res["resposta_gabarito"].tolist(),
            )
            df_res["S_bertscore_p"] = P
            df_res["S_bertscore_r"] = R
            df_res["S_bertscore_f1"] = F1
        except Exception:
            print("BERTScore falhou -- dados brutos salvos, rode 04_metricas.py depois.")
            traceback.print_exc()

    completa = len(concluidas) >= len(df_perguntas)
    if completa:
        comum.salvar_parquet_atomico(df_res, arq_res, permitir_encolher=True)
        comum.limpar_parciais(arq_res, arq_ent)
        print(f"-> {arq_res}")
    else:
        print(f"  [checkpoint] rodada INCOMPLETA "
              f"({len(concluidas)}/{len(df_perguntas)} perguntas).")
        print(f"  O arquivo final NÃO foi gravado de propósito: reexecute o mesmo")
        print(f"  comando e a rodada continua da pergunta {len(concluidas) + 1}.")

    # Item 12 + item 03: registro de ambiente e auditoria do alinhamento.
    try:
        import json as _json
        frac_alinhado = (
            float((df_ent["inicio_janela_entropia"] > 0).mean()) if len(df_ent) else None)
        meta = gerador.metadata({
            "bloco": 2, "retriever": comum.RETRIEVER,
            "embed_model_id": comum.EMBED_MODEL_ID,
            # perguntas, não linhas: com duas modalidades len(df_res) é o dobro
            "n_perguntas": int(df_res["pergunta_idx"].nunique()) if len(df_res) else 0,
            "n_linhas_resultado": int(len(df_res)),
            "k_top": comum.K_TOP,
            "com_sondas": com_sondas,
            "criterios": criterios,
            "prefixo_forcado": prefixo_forcado,
            # se o modelo declara marcadores e isto vier 0, o alinhamento não
            # aconteceu e a entropia mediu a abertura do canal de análise
            "fracao_posicoes_com_inicio_maior_que_zero": frac_alinhado,
            "min_passos_entropia_para_agregar": comum.MIN_PASSOS_ENTROPIA,
            # contadores do gerador cobrem só o que rodou NESTE processo
            "retomado_de_n_perguntas": n_retomadas,
            "n_perguntas_neste_processo": len(concluidas) - n_retomadas,
        })
        with open(os.path.join(comum.DIR_SAIDAS, f"metadata_b2_{nome}.json"),
                  "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)
        if gerador.ids_marcadores and not frac_alinhado:
            print("  [ATENÇÃO] o modelo declara marcadores de fim de prefácio, mas "
                  "NENHUMA posição teve inicio > 0.")
            print("  A janela de entropia pode ter medido o canal de análise. "
                  "Avise o Murilo antes de usar estes números.")
    except Exception:
        traceback.print_exc()


    # ---- resumo em tela ----------------------------------------------
    if len(df_res):
        n_perg = df_res["pergunta_idx"].nunique()
        baseline_acaso = df_res["n_golds_no_top10"].mean() / comum.K_TOP
        print(f"\n=== {tag}: {(time.time() - t_inicio) / 60:.1f} min ===")
        uma = df_res.drop_duplicates("pergunta_idx")
        print(f"  reposicionamento do gold: "
              f"{int(uma['gold_foi_reposicionado_no_top10'].sum())}/{n_perg} perguntas")
        print(f"  acaso esperado (n_golds/K): {baseline_acaso:.1%}")
        print(f"  P@1 do retriever (invariante ao critério e ao modelo): "
              f"{uma['max_similaridade_e_gold'].mean():.1%}")

        for criterio in criterios:
            sub = df_res[df_res["criterio"] == criterio]
            if not len(sub):
                continue
            rot = "A (regex)" if criterio == "regex" else "B (JSON)"
            print(f"\n  --- Modalidade {rot} ---")
            print(f"    posições válidas por pergunta: "
                  f"{sub['n_posicoes_validas'].mean():.1f} de {comum.K_TOP}")
            print(f"    menor entropia é o gold:  {sub['min_entropia_e_gold'].mean():.1%}"
                  f"   [SECUNDÁRIA -- ver item 22]")
            print(f"    D_embaralhado mediano:    {sub['D_embaralhado'].median()}")
            print(f"    concordância D x entropia: "
                  f"{sub['concordancia_D_entropia'].mean():.1%}  (ler contra a nula exata)")
            # a entropia desta modalidade é sinal, ou é a decisão de parada
            # escrita em nats? (ver comum.diagnostico_circularidade)
            se = df_ent[df_ent["criterio"] == criterio] if len(df_ent) else df_ent
            if len(se):
                dc = comum.diagnostico_circularidade(se["entropia_media"], se["valida"])
                if dc["auc"] is not None:
                    print(f"    AUC 'recusa tem entropia menor': {dc['auc']:.3f}"
                          f"{'  [SEPARAÇÃO TOTAL]' if dc['separacao_total'] else ''}")
                    for linha in comum.texto_alerta_circularidade(dc, criterio):
                        print("  " + linha)
            if criterio == "json":
                n_inf = sub["n_inferencias"].sum()
                if n_inf:
                    print(f"    conformidade de formato:  "
                          f"{100 * sub['n_pos_conformes'].sum() / n_inf:.1f}%"
                          f"  (JSON estrito {100 * sub['n_pos_json_estrito'].sum() / n_inf:.1f}%)")
                    frac_jan = 100 * sub["n_pos_janela_json_alinhada"].sum() / n_inf
                    print(f"    janela de entropia alinhada ao valor da chave: {frac_jan:.1f}%")
                    if frac_jan < 80:
                        print("    <<< ATENÇÃO: abaixo de 80%. Nas posições não alinhadas a")
                        print("        entropia mede a sintaxe do JSON, não a resposta, e NÃO")
                        print("        é comparável com a Modalidade A. Filtre por")
                        print("        janela_json_alinhada antes de comparar.")
                div = (sub["D_embaralhado"].fillna(-1)
                       != sub["D_embaralhado_regra_alternativa"].fillna(-1))
                print(f"    D difere sob 'sentinela OU regex no valor': "
                      f"{int(div.sum())}/{len(sub)}")

        # ---- comparação A x B, pareada por pergunta ----
        if len(criterios) > 1:
            piv = df_res.pivot_table(index="pergunta_idx", columns="criterio",
                                     values="D_embaralhado")
            if {"regex", "json"} <= set(piv.columns):
                amb = piv.dropna()
                print("\n  --- Modalidade A x B, pareado por pergunta ---")
                print(f"    perguntas em que AS DUAS pararam: {len(amb)}/{n_perg}")
                if len(amb):
                    print(f"    D igual nas duas: {100 * (amb['regex'] == amb['json']).mean():.1f}%"
                          f"   delta médio (json - regex): "
                          f"{(amb['json'] - amb['regex']).mean():+.2f}")
                pv = df_res.pivot_table(index="pergunta_idx", columns="criterio",
                                        values="n_posicoes_validas")
                if {"regex", "json"} <= set(pv.columns):
                    print(f"    V médio: regex {pv['regex'].mean():.1f}  "
                          f"json {pv['json'].mean():.1f}")

    return df_res


def selecionar(perguntas, limite, espalhar=False):
    """
    Subamostra para teste/piloto.

    O padrão histórico é `head(N)`, e ele tem uma armadilha neste dataset: as
    192 perguntas estão ORDENADAS POR ORIGEM. As 74 primeiras são as
    sintéticas, geradas a partir do próprio chunk -- bem formadas, verbosas e
    com o chunk-ouro quase sempre em 1º no ranking. O resto são perguntas
    reais de usuário: curtas, vagas, com erros de digitação.

    Consequência: um piloto com `head(5)` mede o pedaço mais fácil do
    conjunto e produz D=1 em toda parte, o que faz a comparação
    iterativo x incremental parecer degenerada quando pode não ser (e
    vice-versa). Com --espalhar, a amostra cobre a faixa inteira.
    """
    if not limite or limite >= len(perguntas):
        return perguntas
    if not espalhar:
        return perguntas.head(limite)
    idx = np.linspace(0, len(perguntas) - 1, limite).round().astype(int)
    return perguntas.iloc[np.unique(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="todos",
                    help="tags separadas por vírgula, ou 'todos'")
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--espalhar", action="store_true",
                    help="com --limite N, pega N perguntas IGUALMENTE ESPAÇADAS "
                         "ao longo do conjunto em vez das N primeiras. As 192 "
                         "estão ordenadas por origem (74 sintéticas primeiro, "
                         "depois perguntas reais de usuário, curtas e vagas), "
                         "então a cabeça é o pedaço fácil e um piloto sobre ela "
                         "superestima a taxa de parada precoce")
    ap.add_argument("--smoke", action="store_true",
                    help="usa um LLM falso, sem GPU: valida a lógica do pipeline")
    ap.add_argument("--sem-bertscore", action="store_true")
    ap.add_argument("--criterios", default=",".join(comum.CRITERIOS),
                    help="critério de parada: regex (Modalidade A), json "
                         "(Modalidade B) ou os dois. Padrão: os dois. Cada um "
                         "exige a sua própria geração, então os dois dobram o "
                         "custo do Bloco 2 -- as sondas dos itens 13 e 14 NÃO "
                         "são duplicadas")
    ap.add_argument("--sem-sondas", action="store_true",
                    help="desliga os itens 13 e 14 (teacher forcing e sonda sim/não)")
    ap.add_argument("--prefixo-forcado", nargs="?", const=comum.PREFIXO_FORCADO_PADRAO,
                    default=None,
                    help="item 15: injeta um começo neutro de resposta antes de medir, "
                         "deslocando a janela para depois da bifurcação recusar/responder. "
                         "Dobra o custo do Bloco 2 -- decida depois de olhar o item 06.")
    ap.add_argument("--attn", default=None,
                    help="sobrescreve attn_implementation (ex.: flex_attention, sdpa, "
                         "eager). O gpt-oss:20b já vem com flex_attention por padrão: "
                         "sem isso o braço fixo k=50 não cabe numa 5090")
    ap.add_argument("--quant", default=None, choices=["4bit", "8bit", "nativo"],
                    help="item 17: sobrescreve a quantização do MODELOS")
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--recomecar", action="store_true",
                    help="ignora e apaga o checkpoint parcial deste modelo "
                         "(o padrão é RETOMAR de onde parou)")
    args = ap.parse_args()

    os.makedirs(comum.DIR_SAIDAS, exist_ok=True)
    comum.verificar_corpus_alinhado()

    tags = comum.MODELOS_BLOCO2 if args.modelos == "todos" else [
        t.strip() for t in args.modelos.split(",") if t.strip()
    ]

    criterios = [c.strip() for c in args.criterios.split(",") if c.strip()]
    for c in criterios:
        if c not in comum.CRITERIOS:
            raise SystemExit(f"critério desconhecido: {c} (use {comum.CRITERIOS})")

    corpus = pd.read_parquet(comum.ARQ_CORPUS)
    perguntas = pd.read_parquet(comum.ARQ_PERGUNTAS_74)
    embs = pd.read_parquet(comum.ARQ_EMB_74)
    perguntas = selecionar(perguntas, args.limite, args.espalhar)

    print(f"Corpus: {len(corpus)} chunks | Perguntas: {len(perguntas)} | "
          f"K_TOP={comum.K_TOP} | SEED_BASE={comum.SEED_BASE} | "
          f"N_TOKENS_ENTROPIA={comum.N_TOKENS_ENTROPIA} | criterios={criterios}")
    print(f"gerações por pergunta: {comum.K_TOP} posições x {len(criterios)} critério(s) "
          f"= {comum.K_TOP * len(criterios)}")
    matriz = comum.matriz_corpus(corpus)

    for tag in tags:
        sufixo = "__prefixo" if args.prefixo_forcado else ""
        arq = os.path.join(comum.DIR_SAIDAS,
                           f"b2_resultados_{comum.nome_saida(tag)}{sufixo}.parquet")
        if os.path.exists(arq) and not args.forcar:
            print(f"\n[pular] {tag}: {arq} já existe (use --forcar para refazer)")
            continue
        print("\n" + "=" * 78)
        rodar_modelo(tag, perguntas, embs, corpus, matriz,
                     smoke=args.smoke, com_bertscore=not args.sem_bertscore,
                     com_sondas=not args.sem_sondas,
                     prefixo_forcado=args.prefixo_forcado, quant=args.quant, attn=args.attn,
                     recomecar=args.recomecar, criterios=criterios)

    print("\nBloco 2 concluído. Devolva a pasta saidas/ inteira para o Murilo.")


if __name__ == "__main__":
    main()
