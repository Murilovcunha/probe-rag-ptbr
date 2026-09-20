"""
02_bloco1_desempenho.py -- BLOCO 1: Avaliação de Desempenho
(Iterativo vs. Incremental vs. Contexto Fixo k=50)

Objetivo: medir fidelidade (S) e custo computacional (eta) sob três formas de
injeção de contexto, com o MESMO ranking de recuperação para as três.

Dataset : perguntas_192.parquet (192 questões, dataset híbrido)
Texto   : coluna `texto_pagina` do corpus (páginas inteiras, ~400 tokens)
Ranking : similaridade de cosseno com Qwen3-Embedding-8B, D_max = 50

Os três modos rodam SEQUENCIALMENTE para cada pergunta, com o modelo já
carregado, para que a comparação não sofra variação de estado da GPU:

  1. ITERATIVO (artigo base, 2026.propor-2.14)
     Iteração i recebe APENAS o chunk/página da posição i -- o anterior é
     substituído, não somado. W_i = {c_i}.

  2. INCREMENTAL (tese, Eq. 20 com passo s=1)
     Iteração i recebe o contexto acumulado W_i = c_1 ∪ ... ∪ c_i.

  3. FIXO k=50 (baseline de custo)
     Uma única inferência com as 50 páginas do topo do ranking. Fornece o
     T_fix medido, necessário para o eixo Z da tese: eta = 1 - T_inc/T_fix
     (Eq. 25). Sem ele, o eta não é calculável.

EARLY-STOPPING REAL: nos modos 1 e 2, o laço é interrompido no instante em
que o gerador declara suficiência informacional. Não existe geração
especulativa nem varredura completa "por garantia": o custo medido é o custo
real do método.

DOIS CRITÉRIOS DE PARADA por pergunta (tese, seção 4.3.2). Cada pergunta é
varrida sob as duas modalidades, uma após a outra, com o mesmo ranking e o
mesmo estado de GPU:

  A. "regex"  Autômato de expressões regulares sobre a resposta em prosa
              (Eq. 21). É o critério do artigo base -- prompt congelado.
  B. "json"   Saída estruturada: o gerador devolve {"resposta": "..."} e
              declara insuficiência preenchendo a chave com a string exata
              "NÃO ENCONTRADO" (Eq. 22).

Na Modalidade B, a taxa de conformidade de formato é registrada como métrica
autônoma (conformidade_json / json_estrito), porque falhas de formato se
confundem com falhas de ancoragem na medida de D e precisam ser separadas.
Formato quebrado NÃO trava o laço: cai no autômato da Modalidade A sobre o
texto bruto e a iteração fica marcada com conformidade_json=False.

Total: 2 critérios x 3 modos = 6 varreduras por pergunta.

OTIMIZAÇÃO DE VRAM (obrigatória neste bloco): generate() é chamado com
output_scores=False e return_dict_in_generate=False. No modo incremental o
contexto chega a 50 páginas; materializar a pilha de logits (|vocab| floats
por passo, e o vocabulário destes modelos passa de 250 mil tokens) somaria
dezenas de GB e derrubaria a 5090 sozinha. Logits são extraídos SÓ no
Bloco 2, onde o contexto é de um único chunk.

Saídas (em saidas/):
  b1_resumo_<modelo>.parquet      1 linha por (pergunta, critério, modo)
  b1_iteracoes_<modelo>.parquet   1 linha por (pergunta, critério, modo, iteração)

Uso:
  python 02_bloco1_desempenho.py                       # 6 modelos, 2 critérios
  python 02_bloco1_desempenho.py --modelos gemma3:27b
  python 02_bloco1_desempenho.py --criterios regex     # só a Modalidade A
  python 02_bloco1_desempenho.py --limite 3 --smoke    # teste de fumaça, sem GPU
"""

import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd

import comum

MODOS_PADRAO = ["iterativo", "incremental", "fixo50"]
RESPOSTA_FALLBACK = "A informação não foi encontrada nos documentos disponíveis."


# ============================================================
# Montagem do contexto de uma iteração
# ============================================================

def contexto_da_iteracao(paginas: list[str], modo: str, i: int):
    """
    `paginas` = textos (texto_pagina) das posições 1..D_max do ranking.
    `i` = índice 1-based da iteração.

    iterativo   -> string única (a página da posição i). Passar string faz
                   comum.montar_mensagens usar o prompt de UM documento, que
                   é byte-a-byte o mesmo prompt do artigo base.
    incremental -> lista com as páginas 1..i (contexto cumulativo, Eq. 20).
    """
    if modo == "iterativo":
        return paginas[i - 1]

    acumulado = paginas[:i]
    if comum.EVITAR_REPETIR_PAGINA_NO_CONTEXTO:
        # Não altera D (que continua contando posições do ranking de chunks):
        # apenas não concatena duas vezes um texto idêntico. Como 740 chunks
        # apontam para só 462 páginas distintas, sem isso o modo incremental
        # gastaria tokens reenviando a mesma página várias vezes.
        vistos, unicos = set(), []
        for t in acumulado:
            if t not in vistos:
                vistos.add(t)
                unicos.append(t)
        acumulado = unicos

    # ARTEFATO CORRIGIDO (achado no piloto de 04/09/2026).
    # Pela Eq. 20, W_1 = c_1 nos DOIS modos -- a primeira iteração do
    # incremental é, por definição, idêntica à do iterativo. Mas
    # montar_mensagens usa prompts diferentes para string e para lista
    # ("no documento fornecido" x "nos documentos fornecidos", mais o
    # separador "---"), então os dois braços chegavam ao modelo com prompts
    # 3 tokens diferentes já em i=1, e as respostas divergiam em 8 de 10 casos.
    #
    # Isso é fatal quando D=1 domina: no piloto, D_iter=1 em 30/30 varreduras,
    # e nesse regime a comparação iterativo x incremental estaria medindo
    # singular-vs-plural no prompt, não estratégia de injeção de contexto.
    #
    # Com um documento só, devolvemos a STRING -- prompt bit-a-bit igual ao do
    # iterativo. A partir de dois, a lista e o prompt multi-documento.
    if len(acumulado) == 1:
        return acumulado[0]
    return acumulado


# ============================================================
# Um modo, uma pergunta
# ============================================================

def rodar_modo(gerador, pergunta, paginas, ids_chunk, sims_chunk, modo, d_max,
               criterio="regex"):
    """
    Executa um dos três modos para UMA pergunta, sob UM critério de parada.

    `criterio`:
      "regex" -> Modalidade A (Eq. 21), autômato sobre a resposta em prosa
      "json"  -> Modalidade B (Eq. 22), chave "resposta" == "NÃO ENCONTRADO"

    Retorna (resumo, iteracoes). Convenções de registro:
      D_iter  = iteração de parada (Eq. 23). None = censurado à direita.
      D_chunk = nº de segmentos efetivamente injetados até a parada.
      censurado_direita          -> esgotou D_max sem resposta válida
      censurado_limite_tokens    -> parou porque o prompt da próxima iteração
                                    passaria de limite_tokens_prompt
    Registros censurados NÃO devem ser tratados como D = D_max nas médias
    (ver tese, seção 4.4.1) -- por isso D_iter fica nulo e a flag é separada.

    Na Modalidade B, cada iteração grava ainda:
      conformidade_json        -> a chave "resposta" foi lida?
      json_estrito             -> a saída inteira já era o objeto JSON pedido?
      recusa_textual_no_valor  -> o autômato da Modalidade A dispararia sobre
                                  o valor da chave? (permite recalcular D sob
                                  a regra alternativa na análise, sem GPU)
    """
    iteracoes = []
    t0 = time.time()
    comum.zerar_pico_vram()

    tokens_prompt_acum = 0
    tokens_total_acum = 0
    D_iter = None
    D_chunk = None
    id_chunk_parada = None
    resposta_final = RESPOSTA_FALLBACK
    censurado_limite = False
    truncado_fixo = False
    n_paginas_puladas = 0

    if modo == "fixo50":
        # Uma única inferência com as k=50 páginas do topo.
        contexto = contexto_da_iteracao(paginas, "incremental", min(d_max, len(paginas)))
        # guarda de VRAM: encurta o contexto até caber no limite de tokens
        while not isinstance(contexto, str) and len(contexto) > 1:
            n = gerador.contar_tokens_prompt(
                comum.montar_mensagens(pergunta, contexto, criterio=criterio))
            if n <= gerador.limite_tokens_prompt:
                break
            contexto = contexto[:-1]
            truncado_fixo = True
        r = gerador.gerar(pergunta, contexto, criterio=criterio)
        v = comum.validar_resposta(r["texto_resposta"], criterio)
        valida = v["valida"]
        tokens_prompt_acum = r["n_tokens_prompt"]
        tokens_total_acum = r["n_tokens_prompt"] + r["n_tokens_resposta"]
        iteracoes.append({
            "criterio": criterio,
            "modo": modo, "iteracao": 1, "posicao_ranking": None,
            "id_chunk": None,
            "n_paginas_no_contexto": 1 if isinstance(contexto, str) else len(contexto),
            "valida": valida,
            "conformidade_json": v["conformidade_json"],
            "json_estrito": v["json_estrito"],
            "recusa_textual_no_valor": v["recusa_textual_no_valor"],
            "motivo_decisao": v["motivo_decisao"],
            "n_tokens_prompt": r["n_tokens_prompt"],
            "n_tokens_resposta": r["n_tokens_resposta"],
            "tokens_total_acum": tokens_total_acum,
            "atingiu_max_tokens": r["atingiu_max_tokens"],
            "similaridade_chunk": None,
            "texto_resposta": v["resposta_extraida"],
            "texto_resposta_bruto": r["texto_resposta"],
        })
        if valida:
            D_iter = 1
            D_chunk = 1 if isinstance(contexto, str) else len(contexto)
            resposta_final = v["resposta_extraida"]
        return (
            {
                "criterio": criterio,
                "modo": modo,
                "D_iter": D_iter,
                "D_iter_regra_alternativa": (
                    1 if (valida and not v["recusa_textual_no_valor"]) else None),
                "D_chunk": D_chunk,
                "id_chunk_parada": None,
                "n_inferencias": 1,
                "tokens_prompt_acum": tokens_prompt_acum,
                "tokens_total_acum": tokens_total_acum,
                "resposta_final": resposta_final,
                "censurado_direita": D_iter is None,
                "censurado_limite_tokens": False,
                "contexto_truncado": truncado_fixo,
                "n_paginas_puladas": 0,
                "n_iter_conformes": 1 if v["conformidade_json"] is True else 0,
                "n_iter_json_estrito": 1 if v["json_estrito"] is True else 0,
                "n_iter_formato_invalido": 1 if v["conformidade_json"] is False else 0,
                "n_iter_max_tokens": 1 if r["atingiu_max_tokens"] else 0,
                "similaridade_na_parada": None,
                "n_paginas_enviadas": 1 if isinstance(contexto, str) else len(contexto),
                "tempo_s": time.time() - t0,
                "vram_pico_gb": comum.vram_pico_gb(),
            },
            iteracoes,
        )

    # ---- modos iterativo e incremental, com early-stopping real ----
    for i in range(1, min(d_max, len(paginas)) + 1):
        contexto = contexto_da_iteracao(paginas, modo, i)

        # Guarda de VRAM: no incremental o prompt cresce a cada iteração.
        # Se a PRÓXIMA inferência estouraria o teto, paramos e marcamos o
        # motivo -- é honesto registrar "parou por limite de memória" em vez
        # de fingir que o método varreu as 50 posições.
        n_prompt = gerador.contar_tokens_prompt(
            comum.montar_mensagens(pergunta, contexto, criterio=criterio))
        if n_prompt > gerador.limite_tokens_prompt:
            if modo == "incremental":
                # o contexto só cresce: se estourou agora, estoura em todas as
                # iterações seguintes -> encerra e registra o motivo
                censurado_limite = True
                print(f"      [limite] iteração {i}: prompt de {n_prompt} tokens "
                      f"> {gerador.limite_tokens_prompt}. Interrompendo este modo.")
                break
            # no modo iterativo cada prompt é independente: uma página
            # excepcionalmente longa é PULADA e o laço continua
            n_paginas_puladas += 1
            print(f"      [limite] iteração {i} pulada: página de {n_prompt} tokens "
                  f"> {gerador.limite_tokens_prompt}.")
            continue

        r = gerador.gerar(pergunta, contexto, criterio=criterio)
        v = comum.validar_resposta(r["texto_resposta"], criterio)
        valida = v["valida"]

        tokens_prompt_acum += r["n_tokens_prompt"]
        tokens_total_acum += r["n_tokens_prompt"] + r["n_tokens_resposta"]

        iteracoes.append({
            "criterio": criterio,
            "modo": modo, "iteracao": i, "posicao_ranking": i,
            "id_chunk": int(ids_chunk[i - 1]),
            "n_paginas_no_contexto": (
                1 if isinstance(contexto, str) else len(contexto)),
            # Item 10: sem esta coluna, "o modelo para quando a similaridade
            # cruza X" não é afirmação testável -- só similaridade_top1 chegava
            # ao resumo, e ela é a mesma para todas as iterações da pergunta.
            "similaridade_chunk": float(sims_chunk[i - 1]),
            "valida": valida,
            "conformidade_json": v["conformidade_json"],
            "json_estrito": v["json_estrito"],
            "recusa_textual_no_valor": v["recusa_textual_no_valor"],
            "motivo_decisao": v["motivo_decisao"],
            "n_tokens_prompt": r["n_tokens_prompt"],
            "n_tokens_resposta": r["n_tokens_resposta"],
            "tokens_total_acum": tokens_total_acum,
            "atingiu_max_tokens": r["atingiu_max_tokens"],
            "texto_resposta": v["resposta_extraida"],
            "texto_resposta_bruto": r["texto_resposta"],
        })

        if valida:
            # EARLY-STOPPING REAL: interrompe agora, não gera as demais.
            D_iter = i
            D_chunk = 1 if isinstance(contexto, str) else len(contexto)
            id_chunk_parada = int(ids_chunk[i - 1])
            resposta_final = v["resposta_extraida"]
            break

    # ---- SIMULAÇÃO RETROATIVA DA REGRA ALTERNATIVA (custo zero) ----
    # Na Modalidade B, "valida" segue a Eq. 22 (só a sentinela exata conta
    # como insuficiência). Aqui recalculamos onde o laço TERIA parado se a
    # regra fosse "sentinela OU recusa em prosa dentro do JSON", usando
    # apenas as respostas já geradas. Sem isso, um modelo que recusa em prosa
    # dentro do JSON aparece com D=1 artificialmente baixo, e não haveria
    # como corrigir a leitura sem rodar tudo de novo.
    D_iter_regra_alternativa = None
    for it in iteracoes:
        if it["posicao_ranking"] is None:
            continue
        if it["valida"] and not it["recusa_textual_no_valor"]:
            D_iter_regra_alternativa = it["iteracao"]
            break

    return (
        {
            "criterio": criterio,
            "modo": modo,
            "D_iter": D_iter,
            "D_iter_regra_alternativa": D_iter_regra_alternativa,
            "D_chunk": D_chunk,
            "id_chunk_parada": id_chunk_parada,
            "n_inferencias": len(iteracoes),
            "tokens_prompt_acum": tokens_prompt_acum,
            "tokens_total_acum": tokens_total_acum,
            "resposta_final": resposta_final,
            "censurado_direita": D_iter is None and not censurado_limite,
            "censurado_limite_tokens": censurado_limite,
            "contexto_truncado": False,
            "n_paginas_puladas": n_paginas_puladas,
            # conformidade de formato agregada no modo (métrica autônoma da
            # seção 4.3.2, usada também como covariável nas análises)
            "n_iter_conformes": sum(
                1 for it in iteracoes if it["conformidade_json"] is True),
            "n_iter_json_estrito": sum(
                1 for it in iteracoes if it["json_estrito"] is True),
            "n_iter_formato_invalido": sum(
                1 for it in iteracoes if it["conformidade_json"] is False),
            "n_iter_max_tokens": sum(
                1 for it in iteracoes if it.get("atingiu_max_tokens")),
            "similaridade_na_parada": (
                float(sims_chunk[D_iter - 1]) if D_iter else None),
            "n_paginas_enviadas": (
                iteracoes[-1]["n_paginas_no_contexto"] if iteracoes else 0
            ),
            "tempo_s": time.time() - t0,
            "vram_pico_gb": comum.vram_pico_gb(),
        },
        iteracoes,
    )


# ============================================================
# Uma rodada completa: um modelo x 192 perguntas x 3 modos
# ============================================================

def rodar_modelo(tag, df_perguntas, df_emb, corpus, matriz, modos, d_max,
                 criterios=None, smoke=False, com_bertscore=True,
                 limite_tokens=None, quant=None, attn=None, recomecar=False,
                 coluna_texto="texto_pagina"):
    criterios = criterios or list(comum.CRITERIOS)
    nome = comum.nome_saida(tag)
    arq_resumo = os.path.join(comum.DIR_SAIDAS, f"b1_resumo_{nome}.parquet")
    arq_iter = os.path.join(comum.DIR_SAIDAS, f"b1_iteracoes_{nome}.parquet")

    if smoke:
        from _gerador_falso import GeradorFalso
        gerador = GeradorFalso(tag)
    else:
        gerador = comum.Gerador(tag, quant=quant, attn=attn)
    if limite_tokens:
        # Válvula de escape para OOM: baixe este valor se o modo incremental
        # estourar a VRAM. Fica registrado em censurado_limite_tokens.
        gerador.limite_tokens_prompt = limite_tokens
        print(f"  limite_tokens_prompt sobrescrito para {limite_tokens}")

    # Unidade de contexto do Bloco 1. O padrão é a página inteira
    # (`texto_pagina`, ~416 palavras na mediana); `texto_artigo` é o chunk
    # restrito (~141 palavras), que é o que o Bloco 2 usa. Ver seção 9.24:
    # com páginas, uma só basta em ~95% das perguntas e o eixo D não varia.
    paginas_corpus = corpus[coluna_texto].astype(str).to_numpy()
    ids_corpus = corpus["id"].to_numpy()

    emb_por_idx = dict(zip(df_emb["pergunta_idx"], df_emb["embedding_pergunta"]))

    # ---- retomada por pergunta ---------------------------------------
    # Uma pergunta concluída tem exatamente len(criterios) x len(modos) linhas
    # no resumo. Menos que isso = morreu no meio, e ela é refeita inteira.
    if recomecar:
        linhas_resumo, linhas_iter, concluidas = [], [], set()
        comum.limpar_parciais(arq_resumo, arq_iter)
        print("  [checkpoint] --recomecar: parciais apagados, começando do zero.")
    else:
        linhas_resumo, linhas_iter, concluidas = comum.retomar_checkpoint(
            arq_resumo, arq_iter, linhas_por_pergunta=len(criterios) * len(modos))
    n_retomadas = len(concluidas)

    t_inicio = time.time()
    desde_checkpoint = 0

    def _gravar_parcial(qtd):
        ok1 = comum.salvar_parquet_atomico(
            pd.DataFrame(linhas_resumo), comum.caminho_parcial(arq_resumo))
        comum.salvar_parquet_atomico(
            pd.DataFrame(linhas_iter), comum.caminho_parcial(arq_iter))
        if ok1:
            print(f"    (checkpoint: {qtd} perguntas concluídas neste modelo)")

    try:
        for _, row in df_perguntas.iterrows():
            idx = int(row["pergunta_idx"])
            if idx in concluidas:
                continue
            pergunta = str(row["pergunta"])
            print(f"\n[{tag}] pergunta {idx + 1}/{len(df_perguntas)}: {pergunta[:70]}...")

            ordem, sims = comum.ranquear(emb_por_idx[idx], matriz, k=None)

            if comum.DEDUP_POR_PAGINA:
                vistos, filtrado = set(), []
                for pos in ordem:
                    txt = paginas_corpus[pos]
                    if txt not in vistos:
                        vistos.add(txt)
                        filtrado.append(pos)
                    if len(filtrado) >= d_max:
                        break
                topo = np.array(filtrado)
            else:
                topo = ordem[:d_max]

            paginas = [paginas_corpus[i] for i in topo]
            ids_chunk = [int(ids_corpus[i]) for i in topo]
            sims_topo = [float(s) for s in sims[: len(topo)]]

            # As 2 modalidades x 3 modos rodam TODAS para esta pergunta antes
            # de passar para a próxima. Assim os dois critérios enfrentam o
            # mesmo estado de GPU e o mesmo ranking, e a diferença observada
            # em D é atribuível ao mecanismo de parada -- não a deriva de
            # ambiente entre duas rodadas separadas por horas.
            for criterio in criterios:
                for modo in modos:
                    resumo, iteracoes = rodar_modo(
                        gerador, pergunta, paginas, ids_chunk, sims_topo, modo, d_max,
                        criterio=criterio
                    )
                    resumo.update({
                        "modelo": tag,
                        "retriever": comum.RETRIEVER,
                        "pergunta_idx": idx,
                        "pergunta": pergunta,
                        "resposta_gabarito": str(row.get("resposta_gabarito", "")),
                        "arquivo_fonte": row.get("arquivo_fonte"),
                        "id_chunk_top1": ids_chunk[0],
                        "similaridade_top1": sims_topo[0],
                    })
                    linhas_resumo.append(resumo)
                    for it in iteracoes:
                        it.update({"modelo": tag, "retriever": comum.RETRIEVER,
                                   "pergunta_idx": idx})
                        linhas_iter.append(it)

                    extra = ""
                    if criterio == "json":
                        extra = (f" conf={resumo['n_iter_conformes']}"
                                 f"/{resumo['n_inferencias']}")
                        if resumo["n_iter_formato_invalido"]:
                            extra += f" [FORMATO x{resumo['n_iter_formato_invalido']}]"
                    print(f"    {criterio:<5} {modo:<12} D_iter={resumo['D_iter']} "
                          f"D_chunk={resumo['D_chunk']} "
                          f"infer={resumo['n_inferencias']} "
                          f"T={resumo['tokens_total_acum']} "
                          f"{resumo['tempo_s']:.1f}s "
                          f"vram={resumo['vram_pico_gb']:.1f}GB" + extra
                          + ("  [CENSURADO]" if resumo["censurado_direita"] else "")
                          + ("  [LIMITE]" if resumo["censurado_limite_tokens"] else ""))

            # Checkpoint. A exposição máxima é CHECKPOINT_BLOCO1 x tempo por
            # pergunta -- a ~2,5-5,6 min cada, 5 perguntas são 12-28 min de
            # risco. Gravação atômica e que se recusa a encolher (ver
            # comum.salvar_parquet_atomico).
            concluidas.add(idx)
            desde_checkpoint += 1
            if desde_checkpoint >= comum.CHECKPOINT_BLOCO1:
                _gravar_parcial(len(concluidas))
                desde_checkpoint = 0

    except (Exception, KeyboardInterrupt) as e:
        print(f"\n!!! INTERROMPIDO ({type(e).__name__}). Salvando o que já foi gerado.")
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
    finally:
        # grava o parcial SEMPRE ao sair do laço, inclusive em Ctrl-C: é o que
        # transforma uma interrupção em retomada barata em vez de perda.
        try:
            if linhas_resumo:
                _gravar_parcial(len(concluidas))
        except Exception:
            traceback.print_exc()
        try:
            gerador.descarregar()
        except Exception:
            pass

    df_resumo = pd.DataFrame(linhas_resumo)
    df_iter = pd.DataFrame(linhas_iter)

    # ---- checkpoint ANTES do BERTScore -------------------------------
    df_resumo.to_parquet(arq_resumo.replace(".parquet", "_checkpoint_sem_bertscore.parquet"),
                         index=False)
    comum.salvar_parquet_atomico(df_iter, arq_iter, permitir_encolher=True)

    # ---- BERTScore (eixo S) ------------------------------------------
    if com_bertscore and len(df_resumo):
        print("\nCalculando BERTScore (eixo S)...")
        try:
            P, R, F1 = comum.calcular_bertscore(
                df_resumo["resposta_final"].tolist(),
                df_resumo["resposta_gabarito"].tolist(),
            )
            df_resumo["S_bertscore_p"] = P
            df_resumo["S_bertscore_r"] = R
            df_resumo["S_bertscore_f1"] = F1
        except Exception:
            print("BERTScore falhou -- os dados brutos estão salvos, rode 04_metricas.py depois.")
            traceback.print_exc()

    # O arquivo FINAL só é gravado quando a rodada terminou. Enquanto
    # incompleta, ele NÃO existe -- e é justamente a ausência dele que faz o
    # laço principal reprocessar este modelo no próximo comando, em vez de
    # pular. Os dados parciais ficam nos *_parcial.parquet.
    completa = len(concluidas) >= len(df_perguntas)
    if completa:
        comum.salvar_parquet_atomico(df_resumo, arq_resumo, permitir_encolher=True)
        comum.limpar_parciais(arq_resumo, arq_iter)
        print(f"\n-> {arq_resumo}\n-> {arq_iter}")
    else:
        print(f"\n  [checkpoint] rodada INCOMPLETA "
              f"({len(concluidas)}/{len(df_perguntas)} perguntas).")
        print(f"  O arquivo final NÃO foi gravado de propósito: reexecute o mesmo")
        print(f"  comando e a rodada continua da pergunta {len(concluidas) + 1}.")

    # Item 12: registro de ambiente junto do parquet, em vez de um pip freeze
    # avulso no fim. A tese exige citar como cada número foi produzido.
    try:
        import json as _json
        meta = gerador.metadata({
            "bloco": 1, "retriever": comum.RETRIEVER,
            "embed_model_id": comum.EMBED_MODEL_ID, "criterios": criterios, "modos": modos, "d_max": d_max,
            # condição necessária para reproduzir D e T: muda a unidade de
            # contexto e, com ela, quantas unidades bastam
            "coluna_texto": coluna_texto,
            "n_perguntas": int(df_resumo["pergunta_idx"].nunique()) if len(df_resumo) else 0,
            "dedup_por_pagina": comum.DEDUP_POR_PAGINA,
            "evitar_repetir_pagina": comum.EVITAR_REPETIR_PAGINA_NO_CONTEXTO,
            "json_decisao": comum.JSON_DECISAO,
            # a rodada pode ter sido retomada: os contadores do gerador só
            # cobrem as perguntas processadas NESTE processo
            "retomado_de_n_perguntas": n_retomadas,
            "n_perguntas_neste_processo": len(concluidas) - n_retomadas,
        })
        with open(os.path.join(comum.DIR_SAIDAS, f"metadata_b1_{nome}.json"),
                  "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


    # ---- resumo em tela ----------------------------------------------
    if len(df_resumo):
        print(f"\n=== {tag}: {(time.time() - t_inicio) / 60:.1f} min ===")

        # NÚMERO DE MANCHETE. Se uma fração não trivial das perguntas bate no
        # teto de tokens, o D do modo incremental está truncado por MEMÓRIA, e
        # a comparação iterativo-vs-incremental deixa de ser justa: um dos
        # braços foi interrompido por motivo que não é ancoragem documental.
        inc = df_resumo[df_resumo["modo"] == "incremental"]
        if len(inc):
            taxa = 100 * inc["censurado_limite_tokens"].mean()
            marca = "  <<< ATENÇÃO" if taxa > 5 else ""
            print(f"  censurado_limite_tokens (incremental): {taxa:.1f}% "
                  f"de {len(inc)} varreduras{marca}")
            if taxa > 5:
                print("    O braço incremental foi interrompido por limite de VRAM, não")
                print("    por ancoragem. Reporte esta taxa junto de qualquer comparação")
                print("    iterativo x incremental, ou baixe --limite-tokens e refaça.")
        for criterio in criterios:
            print(f"  --- Modalidade {'A (regex)' if criterio == 'regex' else 'B (JSON)'} ---")
            dfc = df_resumo[df_resumo["criterio"] == criterio]
            for modo in modos:
                sub = dfc[dfc["modo"] == modo]
                if not len(sub):
                    continue
                validas = sub[sub["D_iter"].notna()]
                linha = (f"    {modo:<12} "
                         f"parou em {len(validas)}/{len(sub)}  "
                         f"D_iter mediano="
                         f"{validas['D_iter'].median() if len(validas) else float('nan')}  "
                         f"T médio={sub['tokens_total_acum'].mean():.0f}")
                if "S_bertscore_f1" in sub:
                    linha += f"  S(F1)={sub['S_bertscore_f1'].mean():.4f}"
                print(linha)
            if criterio == "json":
                infer = dfc["n_inferencias"].sum()
                if infer:
                    print(f"    conformidade de formato: "
                          f"{100 * dfc['n_iter_conformes'].sum() / infer:.1f}% das "
                          f"{infer} inferências  "
                          f"(JSON estrito: "
                          f"{100 * dfc['n_iter_json_estrito'].sum() / infer:.1f}%)")
                # divergência entre a Eq. 22 e a regra alternativa
                alt = dfc["D_iter_regra_alternativa"]
                div = int((dfc["D_iter"].fillna(-1) != alt.fillna(-1)).sum())
                print(f"    D difere sob 'sentinela OU regex no valor' em "
                      f"{div}/{len(dfc)} casos (recusa em prosa dentro do JSON)")

            # eta = 1 - T/T_fix (Eq. 25), pareado por pergunta, DENTRO do critério
            if "fixo50" in modos:
                piv = dfc.pivot_table(index="pergunta_idx", columns="modo",
                                      values="tokens_total_acum")
                for m in ("iterativo", "incremental"):
                    if {m, "fixo50"} <= set(piv.columns):
                        eta = 1 - piv[m] / piv["fixo50"]
                        print(f"    eta {m:<12} vs fixo k=50: média {eta.mean():+.3f}  "
                              f"mediana {eta.median():+.3f}  "
                              f"negativo em {int((eta < 0).sum())}/{len(eta)}")

    return df_resumo


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
    ap.add_argument("--modos", default=",".join(MODOS_PADRAO),
                    help="iterativo,incremental,fixo50")
    ap.add_argument("--criterios", default=",".join(comum.CRITERIOS),
                    help="critério de parada: regex (Modalidade A), json "
                         "(Modalidade B) ou os dois. Padrão: os dois, "
                         "sequencialmente para cada pergunta")
    ap.add_argument("--limite", type=int, default=None,
                    help="usar só as N primeiras perguntas (para teste). Veja --espalhar: as primeiras 74 das 192 são as sintéticas, escritas A PARTIR do chunk, e por isso as mais fáceis -- uma amostra de cabeça não representa o conjunto")
    ap.add_argument("--espalhar", action="store_true",
                    help="com --limite N, pega N perguntas IGUALMENTE ESPAÇADAS "
                         "ao longo do conjunto em vez das N primeiras. As 192 "
                         "estão ordenadas por origem (74 sintéticas primeiro, "
                         "depois perguntas reais de usuário, curtas e vagas), "
                         "então a cabeça é o pedaço fácil e um piloto sobre ela "
                         "superestima a taxa de parada precoce")
    ap.add_argument("--coluna-texto", default="texto_pagina",
                    choices=["texto_pagina", "texto_artigo"],
                    help="unidade de contexto. texto_pagina (padrão, ~416 "
                         "palavras) é o desenho original; texto_artigo (~141, "
                         "o mesmo do Bloco 2) torna cada unidade 3x menor e é "
                         "a alavanca mais barata para o eixo D voltar a variar "
                         "-- ver seção 9.24 do README")
    ap.add_argument("--dmax", type=int, default=comum.D_MAX)
    ap.add_argument("--limite-tokens", type=int, default=None,
                    help="sobrescreve o teto de tokens de entrada por inferência "
                         "(baixe para 16384 se o modo incremental der OOM)")
    ap.add_argument("--smoke", action="store_true",
                    help="usa um LLM falso, sem GPU: valida a lógica do pipeline")
    ap.add_argument("--sem-bertscore", action="store_true")
    ap.add_argument("--attn", default=None,
                    help="sobrescreve attn_implementation (ex.: flex_attention, sdpa, "
                         "eager). O gpt-oss:20b já vem com flex_attention por padrão: "
                         "sem isso o braço fixo k=50 não cabe numa 5090")
    ap.add_argument("--quant", default=None, choices=["4bit", "8bit", "nativo"],
                    help="sobrescreve a quantização do MODELOS (item 17): permite "
                         "medir quanto da dispersão entre modelos vem do formato "
                         "de pesos e não da arquitetura")
    ap.add_argument("--forcar", action="store_true",
                    help="reprocessa modelos que já têm arquivo de saída")
    ap.add_argument("--recomecar", action="store_true",
                    help="ignora e apaga o checkpoint parcial deste modelo, "
                         "recomeçando da pergunta 1 (o padrão é RETOMAR)")
    args = ap.parse_args()

    os.makedirs(comum.DIR_SAIDAS, exist_ok=True)
    comum.verificar_corpus_alinhado()

    tags = comum.MODELOS_BLOCO1 if args.modelos == "todos" else [
        t.strip() for t in args.modelos.split(",") if t.strip()
    ]
    modos = [m.strip() for m in args.modos.split(",") if m.strip()]
    for m in modos:
        if m not in MODOS_PADRAO:
            raise SystemExit(f"modo desconhecido: {m} (use {MODOS_PADRAO})")
    criterios = [c.strip() for c in args.criterios.split(",") if c.strip()]
    for c in criterios:
        if c not in comum.CRITERIOS:
            raise SystemExit(f"critério desconhecido: {c} (use {comum.CRITERIOS})")

    corpus = pd.read_parquet(comum.ARQ_CORPUS)
    perguntas = pd.read_parquet(comum.ARQ_PERGUNTAS_192)
    embs = pd.read_parquet(comum.ARQ_EMB_192)
    if args.limite:
        perguntas = selecionar(perguntas, args.limite, args.espalhar)

    print(f"Corpus: {len(corpus)} chunks | Perguntas: {len(perguntas)} | "
          f"D_max={args.dmax} | modos={modos} | criterios={criterios}")
    print(f"DEDUP_POR_PAGINA={comum.DEDUP_POR_PAGINA} | "
          f"EVITAR_REPETIR_PAGINA_NO_CONTEXTO={comum.EVITAR_REPETIR_PAGINA_NO_CONTEXTO} | "
          f"JSON_DECISAO={comum.JSON_DECISAO}")
    print(f"varreduras por pergunta: {len(criterios)} critério(s) x {len(modos)} modo(s) "
          f"= {len(criterios) * len(modos)}")
    matriz = comum.matriz_corpus(corpus)

    for tag in tags:
        arq = os.path.join(comum.DIR_SAIDAS, f"b1_resumo_{comum.nome_saida(tag)}.parquet")
        if os.path.exists(arq) and not args.forcar:
            print(f"\n[pular] {tag}: {arq} já existe (use --forcar para refazer)")
            continue
        print("\n" + "=" * 78)
        rodar_modelo(tag, perguntas, embs, corpus, matriz, modos, args.dmax,
                     coluna_texto=args.coluna_texto,
                     criterios=criterios,
                     smoke=args.smoke, com_bertscore=not args.sem_bertscore,
                     limite_tokens=args.limite_tokens, quant=args.quant, attn=args.attn,
                     recomecar=args.recomecar)

    print("\nBloco 1 concluído. Devolva a pasta saidas/ inteira para o Murilo.")


if __name__ == "__main__":
    main()
