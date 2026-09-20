"""
comum.py -- Módulo compartilhado pelos experimentos do Bloco 1 e do Bloco 2.

Concentra tudo o que os dois blocos têm em comum, para que a única diferença
entre eles seja a DINÂMICA de injeção de contexto (e a extração ou não de
logits):

  * MODELOS         -> mapeamento tag Ollama  ->  repositório Hugging Face
  * Gerador         -> carregamento em 4-bit, chat template, generate(),
                       desligamento do thinking mode e limpeza de VRAM
  * regex de parada -> resposta_indica_nao_encontrado() (idêntico à produção)
  * prompt          -> montar_mensagens() (idêntico ao experimento.py)
  * ranking         -> similaridade de cosseno vetorizada sobre o corpus todo
  * entropia        -> Shannon exata sobre os logits dos primeiros N tokens

Nada aqui carrega modelo na importação: todo custo de GPU é explícito.

Autor: pacote gerado para a tese de Murilo Vargas da Cunha (UFPel).
"""

from __future__ import annotations

import gc
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

# ============================================================
# 1. CONFIGURAÇÃO DOS MODELOS  (tag Ollama -> repositório Hugging Face)
# ============================================================
#
# `classes` é uma LISTA ORDENADA de classes Auto* que serão tentadas em
# sequência. Isso existe porque a família Gemma/Mistral/Qwen mudou de classe
# canônica entre versões do transformers:
#
#   - "multimodal"  -> AutoModelForMultimodalLM   (Gemma 4, Qwen3.8)
#   - "image_text"  -> AutoModelForImageTextToText (Gemma 3, MedGemma,
#                                                   Mistral Small 3.1)
#   - "causal"      -> AutoModelForCausalLM        (gpt-oss, fallback geral)
#
# Se a primeira classe não existir na versão instalada do transformers, ou
# falhar ao carregar o checkpoint, a próxima é tentada automaticamente.
# Assim o pacote não quebra se o colega tiver uma versão diferente da minha.
#
# `quant`:
#   "4bit"   -> bitsandbytes NF4 + double quant, compute dtype bfloat16.
#   "nativo" -> carrega no formato do próprio checkpoint (usado no gpt-oss:20b,
#               que já é distribuído em MXFP4 e ocupa ~13 GB; requantizar um
#               checkpoint MXFP4 com bitsandbytes só degrada a qualidade sem
#               economizar VRAM).
#
# `limite_tokens_prompt`: teto de tokens de ENTRADA por inferência. Serve de
# guarda contra OOM no Bloco 1 incremental, onde o contexto acumulado pode
# chegar a 50 páginas (~50 mil tokens) e o KV-cache estoura os 32 GB da 5090
# muito antes da janela de contexto nominal do modelo.

MODELOS = {
    # ---------------- Bloco 1 e Bloco 2 ----------------
    "gemma3:27b": {
        "hf_id": "google/gemma-3-27b-it",
        "classes": ["image_text", "causal"],
        "quant": "4bit",
        "thinking": False,          # Gemma 3 não tem thinking mode
        "limite_tokens_prompt": 32768,
        "marcadores_fim_prefacio": [],
    },
    "gpt-oss:20b": {
        "hf_id": "openai/gpt-oss-20b",
        "classes": ["causal"],
        "quant": "nativo",          # MXFP4 nativo, ~13 GB -- NÃO requantizar
        "thinking": False,
        "reasoning_effort": "low",  # gpt-oss sempre emite canal de análise;
                                    # "low" encurta esse canal ao mínimo
        "limite_tokens_prompt": 32768,
        # o texto útil do gpt-oss vem depois do último <|message|> (canal final)
        "marcadores_fim_prefacio": ["<|message|>"],
        # Item 14: na sonda sim/não NÃO dá para ler o passo 0 -- ele é marcador
        # de canal do formato harmony, não "Sim"/"Não". Forçamos o cabeçalho do
        # canal final dentro do prompt e lemos o passo seguinte.
        "prefixo_canal_final": "<|channel|>final<|message|>",
        # BACKEND DE ATENÇÃO (rodada de 04/09/2026, RTX 5090).
        # O GptOssForCausalLM usa attention sinks, que o SDPA não implementa;
        # o transformers então cai para `eager`, que materializa a matriz de
        # scores inteira [64 cabeças x seq x seq]. Num prompt de 32.768 tokens
        # isso é 128 GiB só nessa alocação, e o braço fixo k=50 morre na
        # primeira pergunta. O flash-attention não tem kernel para sm_120
        # (sinks só a partir de SM90), então a saída é o flex_attention do
        # PyTorch, que tem memória linear: 16,3 GB medidos a 30k tokens.
        # Diferença numérica da ordem de 1e-3, contra 1e-1 já introduzida pela
        # heterogeneidade MXFP4 vs NF4 que o desenho assume (item 17).
        # Os outros cinco modelos seguem no padrão (None = escolha do
        # transformers, tipicamente SDPA).
        "attn_implementation": "flex_attention",
    },
    "MedAIBase/MedGemma1.0:27b-it-q8_0": {
        # Variante multimodal oficial da MedGemma 1.0 27B instruct.
        # Se preferir a variante SÓ TEXTO (mais leve de carregar, mesmos pesos
        # de linguagem), troque por "google/medgemma-27b-text-it".
        "hf_id": "google/medgemma-27b-it",
        "classes": ["image_text", "causal"],
        "quant": "4bit",
        "thinking": False,
        "limite_tokens_prompt": 32768,
        "marcadores_fim_prefacio": [],
    },
    "mistral-small3.1:latest": {
        "hf_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "classes": ["image_text", "causal"],
        "quant": "4bit",
        "thinking": False,
        "limite_tokens_prompt": 32768,
        "marcadores_fim_prefacio": [],
    },
    # ---------------- Só Bloco 1 ----------------
    "gemma4:31b": {
        "hf_id": "google/gemma-4-31B-it",
        "classes": ["multimodal", "image_text", "causal"],
        "quant": "4bit",
        "thinking": False,          # thinking vem LIGADO por padrão -> desligar
        "limite_tokens_prompt": 32768,
        "marcadores_fim_prefacio": ["</think>", "<end_of_thinking>"],
    },
    "qwen3.8:27b": {
        "hf_id": "Qwen/Qwen3.8-27B",
        "classes": ["multimodal", "causal"],
        "quant": "4bit",
        "thinking": False,          # thinking vem LIGADO por padrão -> desligar
        "limite_tokens_prompt": 32768,
        "marcadores_fim_prefacio": ["</think>"],
    },
}

# Modelos de cada bloco, na ordem de execução sugerida (do mais leve ao mais
# pesado de baixar, para o colega já ter resultado parcial cedo).
MODELOS_BLOCO1 = [
    "gpt-oss:20b",
    "mistral-small3.1:latest",
    "gemma3:27b",
    "MedAIBase/MedGemma1.0:27b-it-q8_0",
    "gemma4:31b",
    "qwen3.8:27b",
]
MODELOS_BLOCO2 = [
    "gpt-oss:20b",
    "mistral-small3.1:latest",
    "gemma3:27b",
    "MedAIBase/MedGemma1.0:27b-it-q8_0",
]

# ------------------------------------------------------------------
# RETRIEVER -- fator experimental desde 06/09/2026 (ver README 9.25)
# ------------------------------------------------------------------
# O componente de recuperação era uma constante do desenho. Deixou de ser:
# com qwen3 o chunk-ouro cai em 1º com frequência tão alta que D=1 em ~95%
# das perguntas e o contraste iterativo x incremental não tem onde
# acontecer. Trocar por um embedder pequeno e real (e5-small) faz D variar
# -- e, mais importante, transforma a qualidade do retriever em FATOR, o
# que responde "quando a construção incremental compensa?" em vez de
# "ela compensa?".
#
# LOGPROB_RETRIEVER escolhe o conjunto. Ele entra no nome de TODO arquivo de
# saída e no metadata: sem isso, rodar o segundo retriever sobrescreveria os
# resultados do primeiro em silêncio.
RETRIEVERS = {
    "qwen3": {
        "embed_model_id": "Qwen/Qwen3-Embedding-8B",
        "sufixo_corpus": "",          # corpus_export.parquet (o original)
        "prefixo_pergunta": "",
        "prefixo_passagem": "",
        "mrr_medio_artigo": 0.8666,   # Tabela 2 do PROPOR 2026
        "freq_mrr1_artigo": 79.73,
    },
    "e5small": {
        "embed_model_id": "intfloat/multilingual-e5-small",
        "sufixo_corpus": "_e5small",  # corpus_export_e5small.parquet
        # ATENÇÃO: os modelos multilingual-e5 são TREINADOS com estes
        # prefixos. Usá-los de um lado só, ou não usá-los, muda a
        # recuperação -- e mudaria para pior por um motivo errado. Os dois
        # lados (pergunta e passagem) têm de seguir a MESMA convenção com
        # que o corpus foi indexado. Se o corpus vier sem prefixo, deixe
        # estes dois vazios.
        # MEDIDO em 06/09/2026 contra a Tabela 2 do artigo, com as perguntas
        # codificadas das duas formas (ver README 9.27):
        #   SEM prefixo -> MRR 0,6528 / MRR=1 56,76%  (artigo: 0,6782 / 52,70%)
        #   COM prefixo -> MRR 0,4956 / MRR=1 40,54%
        # O corpus foi indexado SEM prefixo. Pôr "query: " só de um lado
        # derrubaria o MRR em 0,16 -- exatamente a assimetria que se temia.
        "prefixo_pergunta": "",
        "prefixo_passagem": "",
        "mrr_medio_artigo": 0.6782,
        "freq_mrr1_artigo": 52.70,
    },
}

RETRIEVER = os.environ.get("LOGPROB_RETRIEVER", "qwen3")
if RETRIEVER not in RETRIEVERS:
    raise SystemExit(f"LOGPROB_RETRIEVER desconhecido: {RETRIEVER!r} "
                     f"(use um de {sorted(RETRIEVERS)})")
CFG_RETRIEVER = RETRIEVERS[RETRIEVER]
EMBED_MODEL_ID = CFG_RETRIEVER["embed_model_id"]
PREFIXO_PERGUNTA = CFG_RETRIEVER["prefixo_pergunta"]
PREFIXO_PASSAGEM = CFG_RETRIEVER["prefixo_passagem"]

# sufixo aplicado aos nomes de saída quando NÃO é o retriever de referência
SUFIXO_RETRIEVER = "" if RETRIEVER == "qwen3" else f"__{RETRIEVER}"

# ============================================================
# 2. HIPERPARÂMETROS GLOBAIS
# ============================================================

D_MAX = 50                  # profundidade máxima do loop de busca (Bloco 1)
K_TOP = 10                  # tamanho do conjunto de candidatos (Bloco 2)
SEED_BASE = 42              # seed determinística: seed(pergunta i) = 42 + i
K_FIXO = 50                 # k do baseline de contexto fixo (para T_fix / eta)
# Teto de tokens da RESPOSTA. Era 400, idêntico ao experimento.py -- e a
# primeira rodada completa (gpt-oss + e5small, 192 perguntas) mostrou que 400
# é BINDING no braço fixo50 e quase só nele:
#     regex  fixo50 35,4% das respostas truncadas | iterativo 7,0%
#     json   fixo50 15,1%                         | iterativo 0,4%
# Truncar enviesa o eixo S contra o baseline, e o viés é grande ao lado do
# efeito medido: no regex, S do fixo50 cai de 0,724 para 0,647 nas truncadas
# (-0,077), enquanto a vantagem do iterativo é de +0,016. Isto é, dois terços
# da vantagem aparente vinham do teto, não da qualidade. No json o sinal
# INVERTE: sem as truncadas o fixo50 passa à frente por +0,022.
# Um teto assimétrico entre os braços mede o teto, não o método -- por isso
# ele subiu. Ver README 9.32.
MAX_NEW_TOKENS = int(os.environ.get("LOGPROB_MAX_NEW_TOKENS", "800"))

# ------------------------------------------------------------
# Janela de medição da entropia (itens 04 e 06 da qualificação)
# ------------------------------------------------------------
# JANELA PRIMÁRIA, declarada a priori: média sobre os 20 primeiros passos
# úteis. É a definição do experimento.py e é a que entra no enunciado
# principal. NÃO mexer depois de ver resultado.
N_TOKENS_ENTROPIA = 20
# Registro: guardamos os escalares de TODOS os passos gerados, não só dos 20.
# Isso é literalmente grátis -- a pilha de logits inteira já está materializada
# pelo generate() (ver docstring de gerar_com_entropia) -- e permite mostrar se
# o sinal se sustenta ALÉM do ponto de decisão recusa/resposta, que é o que
# separa a hipótese de Nível 1 de um mero detector de recusa. Janelas mais
# largas são análise SECUNDÁRIA pré-declarada.
REGISTRAR_TODOS_OS_PASSOS = True
# Mínimo de passos para uma posição entrar nas médias agregadas (item 04):
# respostas muito curtas produziriam uma média sobre 2 passos ao lado de
# médias sobre 20.
MIN_PASSOS_ENTROPIA = 5
# Top-k do PRIMEIRO passo útil (item 08): separa "confiante de que a resposta
# está aqui" de "confiante de que vai recusar", que a entropia sozinha não
# distingue (a relação é em U).
TOP_K_PRIMEIRO_PASSO = 5

# Bloco 2 sob a Modalidade B (JSON): ALINHAMENTO DA JANELA DE ENTROPIA.
#
# Sob o contrato JSON, os primeiros tokens gerados são sintaxe -- {"resposta": "
# -- e não a resposta. Medir a entropia ali mediria a confiança do modelo em
# abrir um objeto JSON, que é a mesma coisa em toda pergunta e em todo chunk,
# e destruiria a comparação com a Modalidade A.
#
# Com esta flag, a janela é deslocada para o primeiro token DO VALOR da chave,
# usando a mesma mecânica que já pula o canal de análise do gpt-oss. Aí as duas
# modalidades medem a mesma coisa: os 20 primeiros tokens da RESPOSTA.
#
# Ressalva que fica registrada na tese: mesmo alinhada, a distribuição não é
# idêntica -- sob JSON o modelo já se comprometeu com um formato, e isso
# condiciona o que vem depois. A comparação exata entre as modalidades é a da
# DECISÃO DE PARADA (D_embaralhado); a da entropia é aproximada e deve ser
# lida como tal. A coluna `janela_json_alinhada` diz, posição a posição, se o
# deslocamento foi encontrado.
ALINHAR_JANELA_ENTROPIA_JSON = True

# Item 15 -- condição opcional com prefixo forçado: injeta um começo neutro de
# resposta para que a janela caia DEPOIS da bifurcação recusar/responder.
# Ligada por --prefixo-forcado no 03; dobra o custo do Bloco 2.
PREFIXO_FORCADO_PADRAO = "De acordo com o documento,"

# Item 14 -- sonda sim/não. Variantes de superfície: somamos a massa de
# probabilidade sobre o PRIMEIRO token de cada variante, em vez de apostar em
# um id único ("Não" pode sair partido em dois tokens em alguns tokenizadores).
SONDA_VARIANTES_SIM = ["sim", "Sim", "SIM", " sim", " Sim", "▁Sim", "▁sim"]
SONDA_VARIANTES_NAO = ["não", "Não", "NÃO", " não", " Não", "▁Não", "▁não",
                       "nao", "Nao", " nao", " Nao"]

# Evita repetir literalmente a MESMA página dentro de um contexto acumulado.
# Não altera a contagem de D (que continua sendo posição no ranking de
# chunks): apenas não concatena duas vezes um texto idêntico, o que seria
# desperdício puro de tokens no modo incremental. Ver README, seção "Decisões".
EVITAR_REPETIR_PAGINA_NO_CONTEXTO = True

# Deduplicação do PRÓPRIO RANKING por página. False = fiel ao artigo base
# (2026.propor-2.14), em que D conta posições do ranking de chunks.
DEDUP_POR_PAGINA = False

# ------------------------------------------------------------
# Critérios de interrupção determinística (tese, seção 4.3.2)
# ------------------------------------------------------------
# "regex" -> Modalidade A: autômato de expressões regulares sobre a resposta
#            em prosa (Eq. 21). É o critério do artigo base.
# "json"  -> Modalidade B: o gerador é forçado a emitir {"resposta": "..."} e
#            declara insuficiência preenchendo a chave com a string exata
#            "NÃO ENCONTRADO" (Eq. 22).
CRITERIOS = ["regex", "json"]

# Chave e sentinela do contrato JSON da Modalidade B.
JSON_CHAVE_RESPOSTA = "resposta"
JSON_SENTINELA = "NÃO ENCONTRADO"

# Como a Modalidade B decide V(y_i):
#   "sentinela"          -> FIEL À EQ. 22: só a string exata "NÃO ENCONTRADO"
#                           na chave conta como insuficiência. Uma recusa em
#                           prosa DENTRO do JSON (ex.: {"resposta": "Não há
#                           informações no documento."}) é lida como resposta
#                           válida e o laço PARA ali.
#   "sentinela_ou_regex" -> além da sentinela, aplica o autômato da Modalidade
#                           A sobre o VALOR da chave. Mais robusto, menos fiel.
# O padrão é "sentinela" para não desviar da formalização. Independente da
# escolha, cada iteração grava `recusa_textual_no_valor`, o que permite
# recalcular D sob a regra alternativa NA ANÁLISE, sem tocar na GPU --
# mesmo espírito da simulação retroativa do Bloco 2.
JSON_DECISAO = "sentinela"

BERTSCORE_MODEL_TYPE = "neuralmind/bert-base-portuguese-cased"
BERTSCORE_NUM_LAYERS = 8

# Caminhos padrão (relativos à raiz do pacote)
DIR_DADOS = os.environ.get("LOGPROB_DADOS", "dados")
DIR_SAIDAS = os.environ.get("LOGPROB_SAIDAS", "saidas")

_SUF = CFG_RETRIEVER["sufixo_corpus"]
ARQ_CORPUS = os.path.join(DIR_DADOS, f"corpus_export{_SUF}.parquet")
ARQ_PERGUNTAS_192 = os.path.join(DIR_DADOS, "perguntas_192.parquet")
ARQ_PERGUNTAS_74 = os.path.join(DIR_DADOS, "perguntas_74_gold.parquet")
ARQ_EMB_192 = os.path.join(DIR_DADOS, f"embeddings_perguntas_192{_SUF}.parquet")
ARQ_EMB_74 = os.path.join(DIR_DADOS, f"embeddings_perguntas_74{_SUF}.parquet")


def verificar_corpus_alinhado(colunas=("texto_artigo", "texto_pagina"),
                              abortar: bool = True) -> bool:
    r"""
    Os corpora dos vários retrievers têm de conter O MESMO TEXTO.

    Só o `vector` pode diferir entre eles. Se o texto diferir, `D` muda por
    causa do texto e não do ranking, e o retriever deixa de ser um fator
    isolado -- o experimento inteiro passa a medir duas coisas somadas, sem
    sintoma nenhum no log.

    Isto não é hipotético. A exportação em CSV do corpus e5 perdeu barras
    invertidas em 22 de 740 `texto_artigo` e 82 de 740 `texto_pagina`
    (`D:\BKP` virou `D:BKP`); o parquet distribuído foi montado com os textos
    do corpus de referência justamente por isso. Quem reconstruir o corpus a
    partir do CSV cru reintroduz a divergência.

    Compara o corpus do retriever ATIVO com o de referência (qwen3). Não faz
    nada quando só existe um corpus.
    """
    ref = os.path.join(DIR_DADOS, "corpus_export.parquet")
    if RETRIEVER == "qwen3" or not (os.path.exists(ref) and os.path.exists(ARQ_CORPUS)):
        return True
    a = pd.read_parquet(ref, columns=["id", *colunas]).sort_values("id")
    b = pd.read_parquet(ARQ_CORPUS, columns=["id", *colunas]).sort_values("id")
    problemas = []
    if list(a["id"]) != list(b["id"]):
        problemas.append("os `id` não coincidem")
    else:
        for c in colunas:
            n = int((a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy()).sum())
            if n:
                problemas.append(f"{n}/{len(a)} textos diferentes em `{c}`")
    if not problemas:
        return True
    msg = (f"\nCORPUS DESALINHADO entre '{RETRIEVER}' e a referência qwen3:\n"
           + "\n".join(f"  - {p}" for p in problemas)
           + f"\n\n  {ARQ_CORPUS} deve ter os MESMOS textos de {ref}; só o\n"
             "  `vector` pode mudar. Se este corpus foi reconstruído a partir do\n"
             "  CSV, é quase certo que a exportação perdeu as barras invertidas.\n"
             "  Peça o parquet já corrigido em vez de regerá-lo do CSV.\n"
             "  (Para medir assim mesmo, ciente de que o fator fica confundido:\n"
             "   LOGPROB_IGNORAR_CORPUS_DESALINHADO=1)")
    if abortar and not os.environ.get("LOGPROB_IGNORAR_CORPUS_DESALINHADO"):
        raise SystemExit(msg)
    print(msg)
    return False


def nome_saida(tag: str) -> str:
    """
    Nome-base dos arquivos de saída: slug do modelo + sufixo do retriever.

    O sufixo existe para que rodar um segundo retriever NÃO sobrescreva os
    resultados do primeiro. Sem ele, `LOGPROB_RETRIEVER=e5small` apagaria em
    silêncio 8-18 h de GPU já gastas com o qwen3.
    """
    return slug(tag) + SUFIXO_RETRIEVER


def slug(texto: str) -> str:
    """Transforma uma tag de modelo em algo seguro para nome de arquivo."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()


# ============================================================
# 2b. CHECKPOINT E RETOMADA POR PERGUNTA
# ============================================================
#
# A rodada do Bloco 1 leva 8-18 h por modelo. O modo de falha mais provável
# não é bug de lógica -- é OOM na pergunta 150, queda de energia, ou o
# terminal fechado. Sem retomada por pergunta, reexecutar o comando recomeça
# do zero e joga fora 7-17 h de GPU cujos dados estão intactos no disco.
#
# Intervalo entre gravações: a exposição máxima é (intervalo x tempo por
# pergunta). No Bloco 1, a ~2,5-5,6 min por pergunta, 5 perguntas são 12-28
# min de risco. No Bloco 2, a ~1-1,6 min por pergunta, 10 perguntas são
# 10-16 min. Gravar mais miúdo que isso só custa I/O sem reduzir risco real.
CHECKPOINT_BLOCO1 = 5
CHECKPOINT_BLOCO2 = 10


def caminho_parcial(caminho: str) -> str:
    return caminho.replace(".parquet", "_parcial.parquet")


def _num_linhas_parquet(caminho: str):
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(caminho).num_rows
    except Exception:  # noqa: BLE001
        return None


def salvar_parquet_atomico(df: pd.DataFrame, caminho: str,
                           permitir_encolher: bool = False) -> bool:
    """
    Grava um parquet sem janela de corrupção e sem regredir.

    Dois cuidados, ambos aprendidos do jeito difícil:

    1. ESCRITA ATÔMICA. `to_parquet` direto no destino deixa o arquivo
       truncado se o processo morrer no meio da escrita -- e o momento de
       gravar o checkpoint é exatamente quando a máquina está sob pressão de
       memória. Escrevemos em `.tmp` e usamos os.replace(), que é atômico no
       mesmo sistema de arquivos.

    2. NUNCA ENCOLHER. Se um checkpoint de 180 perguntas existe e alguém
       reexecuta o script do zero, a primeira gravação teria 5 perguntas e
       destruiria as 180 -- o dado que sobreviveu ao crash seria apagado pela
       tentativa de recuperação. Aqui a gravação menor é RECUSADA, com aviso.
       Use --recomecar para zerar de propósito.
    """
    if not permitir_encolher and os.path.exists(caminho):
        n_antigo = _num_linhas_parquet(caminho)
        if n_antigo is not None and len(df) < n_antigo:
            print(f"  [checkpoint] recusando gravar {len(df)} linhas sobre "
                  f"{n_antigo} já existentes em {os.path.basename(caminho)}. "
                  f"Use --recomecar se a intenção é zerar.")
            return False
    tmp = caminho + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, caminho)
    return True


def retomar_checkpoint(arq_principal: str, arq_detalhe: str,
                       linhas_por_pergunta: int, chave: str = "pergunta_idx"):
    """
    Lê os parciais e devolve (registros_principal, registros_detalhe,
    perguntas_concluidas).

    Uma pergunta só conta como CONCLUÍDA se tiver exatamente o número
    esperado de linhas no arquivo principal (no Bloco 1, n_critérios x
    n_modos; no Bloco 2, 1). Se o processo morreu no meio de uma pergunta,
    as linhas parciais dela são DESCARTADAS e ela é refeita inteira -- meia
    pergunta no parquet seria pior do que nenhuma, porque entraria nas
    médias como se fosse uma varredura completa.
    """
    p_princ, p_det = caminho_parcial(arq_principal), caminho_parcial(arq_detalhe)
    if not os.path.exists(p_princ):
        return [], [], set()

    try:
        df_p = pd.read_parquet(p_princ)
        df_d = pd.read_parquet(p_det) if os.path.exists(p_det) else pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        print(f"  [checkpoint] parcial ilegível ({type(e).__name__}: {e}); recomeçando.")
        return [], [], set()

    if not len(df_p) or chave not in df_p.columns:
        return [], [], set()

    contagem = df_p.groupby(chave).size()
    completas = set(int(k) for k, v in contagem.items() if v == linhas_por_pergunta)
    incompletas = set(int(k) for k in contagem.index) - completas

    df_p = df_p[df_p[chave].isin(completas)]
    if len(df_d) and chave in df_d.columns:
        df_d = df_d[df_d[chave].isin(completas)]

    print(f"  [checkpoint] retomando: {len(completas)} perguntas já concluídas "
          f"({len(df_p)} linhas)")
    if incompletas:
        print(f"  [checkpoint] {len(incompletas)} pergunta(s) estavam pela metade "
              f"e serão refeitas: {sorted(incompletas)}")
    return (df_p.to_dict("records"),
            df_d.to_dict("records") if len(df_d) else [],
            completas)


def limpar_parciais(*caminhos):
    """Remove os parciais depois que o arquivo final foi gravado com sucesso."""
    for c in caminhos:
        p = caminho_parcial(c)
        for alvo in (p, p + ".tmp"):
            try:
                if os.path.exists(alvo):
                    os.remove(alvo)
            except Exception:  # noqa: BLE001
                pass


# ============================================================
# 3. AUTÔMATO REGEX DE "NÃO ENCONTRADO"
#    Idêntico ao script de produção / experimento.py. NÃO ALTERAR:
#    é o critério de parada V(y_i) da Equação 21 da tese, e mudá-lo
#    invalida a comparação com os resultados já publicados.
# ============================================================

_PADRAO_NAO_ENCONTRADO = (
    r"(não há informações disponíveis"
    r"|não há informações\b"
    r"|não há informação\b"
    r"|não há referência"
    r"|não há instruções"
    r"|não há menção"
    r"|não menciona"
    r"|não foi possível localizar"
    r"|a informação não foi encontrada"
    r"|informação não encontrada"
    r"|não foi encontrad[oa]s?"
    r"|não está presente nos documentos fornecidos"
    r"|não está contida nos documentos fornecidos"
    r"|não encontrei (nenhuma |a )?informaç(ão|ões))"
    r"|documentos?\s+n[aã]o\s+especifica(m)?"
    r"|n[aã]o\s+h[aá]\s+informa[cç][aã]o"
    r"|n[aã]o\s+encontrei\s+nenhuma\s+informa[cç][aã]o"
)


def resposta_indica_nao_encontrado(texto: str | None) -> bool:
    """V(y) = False  <=>  esta função retorna True (Eq. 21 da tese)."""
    if not texto:
        return True
    return re.search(_PADRAO_NAO_ENCONTRADO, texto.lower()) is not None


# ============================================================
# 4. PROMPTS
# ============================================================

_INSTRUCOES_JSON = (
    'Responda SOMENTE com um objeto JSON válido, sem nenhum texto antes ou\n'
    'depois e sem cercas de código, exatamente neste formato:\n'
    '{"' + JSON_CHAVE_RESPOSTA + '": "<sua resposta em português>"}\n'
)


def montar_mensagens(query: str, textos_contexto, criterio: str = "regex") -> list[dict]:
    """
    Monta a lista de mensagens no formato chat.

    `criterio` seleciona a modalidade de interrupção (tese, seção 4.3.2) e,
    com ela, o prompt:
      "regex" -> Modalidade A. Prompt em prosa, byte-a-byte igual ao artigo
                 base e ao experimento.py. NÃO ALTERAR.
      "json"  -> Modalidade B. Mesmas regras de conteúdo, mas o gerador é
                 instruído a devolver {"resposta": "..."} e a preencher a
                 chave com a string exata "NÃO ENCONTRADO" quando o contexto
                 for insuficiente.

    `textos_contexto` pode ser:
      - uma string  -> prompt de UM documento (idêntico ao experimento.py,
                       usado no Bloco 2 e no modo iterativo do Bloco 1);
      - uma lista   -> prompt de N documentos, cada um delimitado por
                       INÍCIO/FIM DO DOCUMENTO (usado no modo incremental e
                       no baseline de contexto fixo do Bloco 1).

    O texto do prompt para 1 documento é byte-a-byte o mesmo do
    experimento.py, para que a entropia do Bloco 2 permaneça comparável.
    """
    if criterio not in CRITERIOS:
        raise ValueError(f"criterio deve ser um de {CRITERIOS}, recebi {criterio!r}")

    um_documento = isinstance(textos_contexto, str)

    if criterio == "regex":
        # ---- Modalidade A: prosa. Texto congelado, não mexer. ----
        if um_documento:
            conteudo = f"""Você é um assistente que responde perguntas com base EXCLUSIVAMENTE
no documento fornecido abaixo.

- Responda em português.
- Seja claro, direto e objetivo.
- NÃO cite fontes.
- NÃO mencione nomes de arquivos.
- Se a resposta não estiver contida no documento, diga explicitamente
  que a informação não foi encontrada.

**Documento para Análise:**
INÍCIO DO DOCUMENTO:
{textos_contexto}
FIM DO DOCUMENTO

**Pergunta do Usuário:**
{query}
"""
            return [{"role": "user", "content": conteudo}]

        blocos = "".join(
            f"INÍCIO DO DOCUMENTO:\n{t}\nFIM DO DOCUMENTO\n\n---\n\n"
            for t in textos_contexto
        )
        conteudo = f"""Você é um assistente que responde perguntas com base EXCLUSIVAMENTE
nos documentos fornecidos abaixo.

- Responda em português.
- Seja claro, direto e objetivo.
- NÃO cite fontes.
- NÃO mencione nomes de arquivos.
- Se a resposta não estiver contida nos documentos, diga explicitamente
  que a informação não foi encontrada.

**Documentos para Análise:**
{blocos}
**Pergunta do Usuário:**
{query}
"""
        return [{"role": "user", "content": conteudo}]

    # ---- Modalidade B: saída estruturada em JSON ----
    # As regras de CONTEÚDO são as mesmas da Modalidade A, de propósito: a
    # única variável manipulada entre os dois critérios é o contrato de
    # saída. Se o texto das instruções divergisse, a diferença observada em
    # D não seria atribuível ao mecanismo de parada.
    doc_sing = "documento" if um_documento else "documentos"
    if um_documento:
        blocos = f"INÍCIO DO DOCUMENTO:\n{textos_contexto}\nFIM DO DOCUMENTO\n"
        rotulo = "**Documento para Análise:**"
    else:
        blocos = "".join(
            f"INÍCIO DO DOCUMENTO:\n{t}\nFIM DO DOCUMENTO\n\n---\n\n"
            for t in textos_contexto
        )
        rotulo = "**Documentos para Análise:**"

    exemplo_sentinela = '{"' + JSON_CHAVE_RESPOSTA + '": "' + JSON_SENTINELA + '"}'
    conteudo = f"""Você é um assistente que responde perguntas com base EXCLUSIVAMENTE
{'no' if um_documento else 'nos'} {doc_sing} fornecido{'' if um_documento else 's'} abaixo.

{_INSTRUCOES_JSON}
- Responda em português.
- Seja claro, direto e objetivo.
- NÃO cite fontes.
- NÃO mencione nomes de arquivos.
- Se a resposta não estiver contida {'no' if um_documento else 'nos'} {doc_sing}, preencha a chave
  "{JSON_CHAVE_RESPOSTA}" com a string EXATA {JSON_SENTINELA}, assim:
  {exemplo_sentinela}
  Não escreva nenhuma explicação nesse caso.

{rotulo}
{blocos}
**Pergunta do Usuário:**
{query}
"""
    return [{"role": "user", "content": conteudo}]


# ============================================================
# 4b. VALIDAÇÃO V(y) NAS DUAS MODALIDADES  (tese, seção 4.3.2)
# ============================================================

def _sem_acento_maiusc(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip().upper()


_SENTINELA_NORM = _sem_acento_maiusc(JSON_SENTINELA)


def extrair_json_resposta(texto: str):
    """
    Tenta obter o valor da chave "resposta" de uma saída da Modalidade B.

    Devolve (valor, conforme, estrito):
      valor   -- conteúdo da chave, ou None se não deu para extrair
      conforme-- True se algum caminho conseguiu ler a chave
      estrito -- True apenas quando a saída INTEIRA já era o objeto JSON
                 pedido, sem cerca de código nem texto ao redor

    Os três caminhos, do mais estrito ao mais tolerante:
      1. json.loads() na saída inteira;
      2. json.loads() no primeiro objeto {...} balanceado (resolve cerca de
         código ```json e frases do tipo "Aqui está: {...}");
      3. regex sobre a chave solta (resolve JSON quebrado por vírgula extra,
         aspas simples ou chave final faltando -- caso comum quando o modelo
         atinge max_new_tokens no meio do objeto).
    A distinção conforme/estrito é o que sustenta a "taxa de conformidade de
    formato" como métrica autônoma exigida pela seção 4.3.2.
    """
    if not texto:
        return None, False, False
    t = texto.strip()

    # 1. saída inteira é o JSON pedido
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and JSON_CHAVE_RESPOSTA in obj:
            return str(obj[JSON_CHAVE_RESPOSTA]), True, True
    except Exception:  # noqa: BLE001
        pass

    # 2. primeiro objeto {...} balanceado dentro do texto
    inicio = t.find("{")
    if inicio >= 0:
        profundidade = 0
        for i in range(inicio, len(t)):
            if t[i] == "{":
                profundidade += 1
            elif t[i] == "}":
                profundidade -= 1
                if profundidade == 0:
                    try:
                        obj = json.loads(t[inicio:i + 1])
                        if isinstance(obj, dict) and JSON_CHAVE_RESPOSTA in obj:
                            return str(obj[JSON_CHAVE_RESPOSTA]), True, False
                    except Exception:  # noqa: BLE001
                        pass
                    break

    # 3. chave solta, tolerante a JSON malformado ou truncado
    m = re.search(
        rf'["\']{JSON_CHAVE_RESPOSTA}["\']\s*:\s*["\'](.*?)["\']\s*[,}}]',
        t, re.DOTALL,
    )
    if not m:
        m = re.search(rf'["\']{JSON_CHAVE_RESPOSTA}["\']\s*:\s*["\'](.*)', t, re.DOTALL)
    if m:
        return m.group(1).strip(), True, False

    return None, False, False


def validar_resposta(texto: str | None, criterio: str = "regex") -> dict:
    """
    Aplica V(y_i) e devolve tudo o que a análise precisa saber sobre a decisão.

    Campos:
      valida                 -- V(y_i): True = o gerador declarou suficiência
                                e o laço PARA aqui
      resposta_extraida      -- texto que vale como resposta (na Modalidade B,
                                o valor da chave; na A, a saída em prosa)
      conformidade_json      -- Modalidade B: a chave foi lida?
      json_estrito           -- Modalidade B: a saída inteira já era o objeto?
      recusa_textual_no_valor-- o autômato da Modalidade A dispara sobre a
                                resposta extraída? Gravado SEMPRE, mesmo
                                quando não é o critério de decisão, para
                                permitir recalcular D na análise sob a outra
                                regra sem tocar na GPU
      motivo_decisao         -- qual caminho decidiu (auditoria)
    """
    if criterio == "regex":
        recusa = resposta_indica_nao_encontrado(texto)
        return {
            "valida": not recusa,
            "resposta_extraida": (texto or "").strip(),
            "conformidade_json": None,
            "json_estrito": None,
            "recusa_textual_no_valor": recusa,
            "motivo_decisao": "regex",
        }

    if criterio != "json":
        raise ValueError(f"criterio desconhecido: {criterio!r}")

    valor, conforme, estrito = extrair_json_resposta(texto)

    if not conforme:
        # FALLBACK: formato quebrado não trava o laço. Aplicamos a Modalidade
        # A sobre o texto bruto e marcamos conformidade_json=False, para que a
        # análise possa filtrar esses casos -- é o que separa "falha de
        # formato" de "falha de ancoragem" na medida de D.
        recusa = resposta_indica_nao_encontrado(texto)
        return {
            "valida": not recusa,
            "resposta_extraida": (texto or "").strip(),
            "conformidade_json": False,
            "json_estrito": False,
            "recusa_textual_no_valor": recusa,
            "motivo_decisao": "fallback_regex_formato_invalido",
        }

    sentinela = _sem_acento_maiusc(valor) == _SENTINELA_NORM
    recusa_textual = resposta_indica_nao_encontrado(valor)

    if JSON_DECISAO == "sentinela":
        valida = not sentinela
        motivo = "json_sentinela"
    elif JSON_DECISAO == "sentinela_ou_regex":
        valida = not (sentinela or recusa_textual)
        motivo = "json_sentinela_ou_regex"
    else:
        raise ValueError(f"JSON_DECISAO inválido: {JSON_DECISAO!r}")

    return {
        "valida": valida,
        "resposta_extraida": valor.strip(),
        "conformidade_json": True,
        "json_estrito": estrito,
        "recusa_textual_no_valor": recusa_textual,
        "motivo_decisao": motivo,
    }


# ============================================================
# 5. RANKING POR SIMILARIDADE DE COSSENO (vetorizado)
# ============================================================

def matriz_corpus(df_corpus: pd.DataFrame, coluna: str = "vector") -> np.ndarray:
    """
    Empilha os embeddings do corpus em uma matriz (n_chunks, dim) já
    L2-normalizada. 740 x 4096 float32 = ~12 MB: cabe folgado na RAM e o
    ranking sai em microssegundos, então NUNCA vale recalcular isso dentro
    do loop de perguntas.
    """
    matriz = np.stack([_para_vetor(v) for v in df_corpus[coluna]]).astype(np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-12
    return matriz / normas


def _para_vetor(x) -> np.ndarray:
    if isinstance(x, str):
        # o CSV original guarda o embedding como texto JSON "[-0.003, 0.004, ...]"
        x = json.loads(x)
    return np.asarray(x, dtype=np.float32)


def ranquear(emb_pergunta, matriz_norm: np.ndarray, k: int | None = None):
    """
    Retorna (indices_ordenados, similaridades_ordenadas) sobre o corpus todo.
    k=None -> corpus inteiro (necessário no Bloco 2 para achar a posição real
    do chunk-ouro mesmo quando ele cai fora do top-100).
    """
    q = _para_vetor(emb_pergunta)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = matriz_norm @ q
    ordem = np.argsort(-sims)
    if k is not None:
        ordem = ordem[:k]
    return ordem, sims[ordem]


# ============================================================
# 6. GESTÃO DE VRAM
# ============================================================

def limpar_vram(*tensores):
    """
    Libera explicitamente os tensores passados e devolve a memória ao driver.

    Chamar DEPOIS de cada inferência. A ordem importa: primeiro solta as
    referências Python (del), depois roda o coletor de ciclos (gc.collect,
    necessário porque os objetos de saída do generate() formam ciclos de
    referência) e só então devolve o cache do allocator ao driver CUDA.
    Sem o gc.collect() no meio, o empty_cache() não encontra nada para
    liberar e a VRAM cresce a cada pergunta até o OOM.
    """
    for t in tensores:
        try:
            del t
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_pico_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def zerar_pico_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 7. GERADOR: carregamento, chat template, generate, entropia
# ============================================================

def _normalizar_entrada(saida):
    """
    Normaliza o retorno de apply_chat_template para um dict de verdade.

    ARMADILHA (encontrada na rodada de 04/09/2026, por Bruno): o transformers
    devolve `BatchEncoding`, que herda de `collections.UserDict` e **não** é
    subclasse de `dict`. Um teste `isinstance(saida, dict)` dá False e embrulha
    o BatchEncoding inteiro dentro de {"input_ids": <BatchEncoding>}, quebrando
    todos os consumidores com `KeyError: 'shape'` e `KeyError: 'input_ids'`.

    Isso NÃO é uma incompatibilidade da série 5 do transformers -- BatchEncoding
    sempre foi UserDict, então o defeito valia para qualquer versão. Ele passou
    pelo --smoke porque o gerador falso não usa chat template nenhum; quem pegou
    foi o piloto do §G, que é justamente para isso.

    Aceita três formas: dict, qualquer mapeamento com .keys() (UserDict,
    BatchEncoding) e o tensor cru de templates que ignoram return_dict.
    """
    if isinstance(saida, dict):
        return saida
    if hasattr(saida, "keys"):          # UserDict / BatchEncoding
        return dict(saida)
    return {"input_ids": saida}         # template devolveu só o tensor


def _auto_classe(nome: str):
    """Resolve o nome curto da classe para a classe Auto* do transformers."""
    import transformers

    mapa = {
        "multimodal": "AutoModelForMultimodalLM",
        "image_text": "AutoModelForImageTextToText",
        "causal": "AutoModelForCausalLM",
    }
    return getattr(transformers, mapa[nome], None)


class Gerador:
    """
    Encapsula tokenizer/processor + modelo + generate().

    Toda a variabilidade entre as seis arquiteturas (classe Auto*, processor
    vs tokenizer, formato do content da mensagem, flag de thinking,
    reasoning_effort, canal de resposta final) fica confinada aqui, para que
    os scripts de experimento sejam idênticos entre modelos.
    """

    def __init__(self, tag_modelo: str, verboso: bool = True, quant: str | None = None,
                 attn: str | None = None):
        from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        if tag_modelo not in MODELOS:
            raise KeyError(
                f"Modelo '{tag_modelo}' não está em MODELOS. "
                f"Disponíveis: {list(MODELOS)}"
            )
        self.tag = tag_modelo
        self.cfg = dict(MODELOS[tag_modelo])
        # Item 17: override de quantização por linha de comando. A entropia é
        # calculada sobre logits e logits mudam com a quantização; poder rodar
        # a mesma pergunta em 4-bit e em 8-bit calibra a magnitude desse efeito
        # e permite reportar a exceção do gpt-oss (MXFP4 nativo) com um número
        # ao lado, em vez de como ressalva qualitativa.
        if quant:
            if quant not in ("4bit", "8bit", "nativo"):
                raise ValueError(f"quant deve ser 4bit, 8bit ou nativo; recebi {quant!r}")
            self.cfg["quant"] = quant
        # Backend de atenção: config do modelo, sobrescritível por --attn.
        # Registramos PEDIDA e EFETIVA separadamente porque o transformers pode
        # recusar silenciosamente a implementação pedida e cair para outra --
        # e é exatamente esse silêncio que derruba o braço fixo k=50.
        self.attn_implementation_pedida = attn or self.cfg.get("attn_implementation")
        self.hf_id = self.cfg["hf_id"]
        self.quant = self.cfg["quant"]
        self.limite_tokens_prompt = self.cfg.get("limite_tokens_prompt", 32768)
        self._aviso_template = False
        # contadores do invariante de índice de logit (§G da qualificação)
        self.n_passos_verificados = 0
        self.n_divergencias_argmax = 0

        if verboso:
            print(f"[carregar] {self.tag}  ->  {self.hf_id}  (quant={self.quant})")

        # ---- tokenizer / processor -------------------------------------
        # Modelos multimodais (Gemma 3/4, Mistral Small 3.1, Qwen3.8) expõem o
        # chat template no PROCESSOR; modelos de texto puro, no TOKENIZER.
        # Tentamos o processor primeiro e caímos para o tokenizer.
        self.proc = None
        try:
            self.proc = AutoProcessor.from_pretrained(self.hf_id)
        except Exception as e:  # noqa: BLE001
            if verboso:
                print(f"  AutoProcessor indisponível ({type(e).__name__}); usando AutoTokenizer.")
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        # objeto que sabe aplicar o chat template
        self.template = self.proc if self.proc is not None else self.tok

        # ---- quantização ----------------------------------------------
        kwargs = {"device_map": "auto"}
        if self.quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            # "nativo": respeita o formato do checkpoint (MXFP4 no gpt-oss)
            kwargs["torch_dtype"] = "auto"

        if self.attn_implementation_pedida:
            kwargs["attn_implementation"] = self.attn_implementation_pedida

        # ---- modelo: tenta as classes na ordem configurada -------------
        ultimo_erro = None
        self.model = None
        for nome_classe in self.cfg["classes"]:
            classe = _auto_classe(nome_classe)
            if classe is None:
                if verboso:
                    print(f"  classe {nome_classe} não existe neste transformers; próxima.")
                continue
            try:
                self.model = classe.from_pretrained(self.hf_id, **kwargs)
                if verboso:
                    print(f"  carregado com {classe.__name__}.")
                break
            except Exception as e:  # noqa: BLE001
                ultimo_erro = e
                if verboso:
                    print(f"  {classe.__name__} falhou ({type(e).__name__}: {e}); próxima.")
        if self.model is None:
            raise RuntimeError(
                f"Nenhuma classe Auto* conseguiu carregar {self.hf_id}. "
                f"Último erro: {ultimo_erro}"
            ) from ultimo_erro

        self.model.eval()

        # ATENÇÃO AO MOMENTO DA CAPTURA (achado do Bruno na rodada de
        # 04/09/2026, generalizado). Os scripts 02 e 03 chamam metadata() DEPOIS
        # de descarregar(), quando self.model já é None. Qualquer campo lido de
        # self.model lá dentro sai vazio -- e como a leitura estava num
        # try/except, saía vazio EM SILÊNCIO. O generation_config_efetivo,
        # justamente o campo que o item 01 precisa para saber se há
        # repetition_penalty ou min_new_tokens interferindo nos logits, vinha {}
        # em todos os metadata_*.json gerados até aqui. Tudo o que depende do
        # modelo carregado é capturado AGORA, na carga.
        self.generation_config_efetivo = self._capturar_generation_config()
        self.attn_implementation_efetiva = (
            getattr(self.model.config, "_attn_implementation", None)
            or getattr(self.model.config, "attn_implementation", None))
        if (self.attn_implementation_pedida
                and self.attn_implementation_efetiva
                and self.attn_implementation_efetiva != self.attn_implementation_pedida):
            print(f"  [ATENÇÃO] pedi attn_implementation="
                  f"{self.attn_implementation_pedida!r} mas o modelo está usando "
                  f"{self.attn_implementation_efetiva!r}. Em prompts longos isso "
                  f"pode estourar a VRAM -- rode verificar_memoria.py antes.")
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id

        # ---- marcadores de fim de prefácio (item 03) -------------------
        # Alinham a janela de entropia ao início da RESPOSTA, pulando o canal
        # de análise do gpt-oss / blocos <think>.
        self.ids_marcadores = []
        marcadores = self.cfg.get("marcadores_fim_prefacio", [])
        for marcador in marcadores:
            mid = self.tok.convert_tokens_to_ids(marcador)
            if not (isinstance(mid, int) and mid >= 0
                    and mid != getattr(self.tok, "unk_token_id", None)):
                # fallback: alguns tokenizadores só resolvem via encode
                ids = self.tok.encode(marcador, add_special_tokens=False)
                mid = ids[0] if len(ids) == 1 else None
            if isinstance(mid, int) and mid >= 0:
                self.ids_marcadores.append(mid)

        # ABORTAR em vez de degradar em silêncio. Se o modelo DECLARA
        # marcadores e nenhum resolveu, `_inicio_janela_entropia` devolveria 0
        # e a entropia mediria a abertura do canal de análise -- com exatamente
        # o mesmo aspecto, no log, de uma rodada bem-sucedida.
        if marcadores and not self.ids_marcadores:
            raise RuntimeError(
                f"{self.hf_id} declara marcadores_fim_prefacio={marcadores}, mas "
                f"nenhum resolveu para um id de token único neste tokenizador. "
                f"A janela de entropia mediria o canal de análise sem avisar. "
                f"Corrija a lista em MODELOS antes de rodar."
            )

        # vocab_size para o registro de ambiente (item 02, reduzido -> item 12)
        try:
            self.vocab_size = int(getattr(self.model.config, "vocab_size", 0)) or len(self.tok)
        except Exception:  # noqa: BLE001
            self.vocab_size = len(self.tok)

        if verboso:
            print(f"  attn: pedida={self.attn_implementation_pedida} "
                  f"efetiva={self.attn_implementation_efetiva}")
            print(f"  vocab_size={self.vocab_size} | marcadores resolvidos: "
                  f"{self.ids_marcadores or 'nenhum (modelo não declara)'}")
            print(f"  pronto. VRAM alocada: {vram_pico_gb():.1f} GB")

    def _capturar_generation_config(self) -> dict:
        """
        Lê do modelo AINDA CARREGADO os parâmetros de geração que podem
        interferir na medida. Os três primeiros são o motivo do item 01
        (`output_logits` em vez de `output_scores`): eles continuam sendo
        aplicados mesmo com do_sample=False, e `min_new_tokens` joga -inf no
        EOS exatamente nos primeiros passos, que é a janela da entropia.
        """
        chaves = ("temperature", "top_p", "top_k", "repetition_penalty",
                  "no_repeat_ngram_size", "min_new_tokens", "min_length",
                  "do_sample", "max_new_tokens", "eos_token_id", "pad_token_id")
        try:
            d = self.model.generation_config.to_dict()
            return {k: d[k] for k in chaves if k in d}
        except Exception as e:  # noqa: BLE001
            # falha registrada, não engolida: um metadata sem generation_config
            # é um metadata que não serve para auditar a medida
            return {"_erro_ao_capturar": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # chat template
    # ------------------------------------------------------------------
    def _extras_template(self):
        extras = {}
        if self.cfg.get("thinking") is False:
            extras["enable_thinking"] = False
        if "reasoning_effort" in self.cfg:
            extras["reasoning_effort"] = self.cfg["reasoning_effort"]
        return extras

    def _renderizar_texto(self, mensagens) -> str:
        """Renderiza o chat template como TEXTO (tokenize=False)."""
        extras = self._extras_template()
        msgs_bloco = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in mensagens
        ]
        ultimo_erro = None
        for msgs, kw in [(msgs_bloco, dict(extras)), (mensagens, dict(extras)),
                         (msgs_bloco, {}), (mensagens, {})]:
            try:
                return self.template.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, **kw)
            except Exception as e:  # noqa: BLE001
                ultimo_erro = e
        raise RuntimeError(f"render do template falhou para {self.hf_id}: {ultimo_erro}")

    def _aplicar_template(self, mensagens, prefixo_forcado: str | None = None):
        """
        Aplica o chat template lidando com as três incompatibilidades comuns:
          1. processors multimodais exigem content como lista de blocos
             [{"type": "text", "text": ...}] em vez de string;
          2. só alguns templates aceitam enable_thinking / reasoning_effort;
          3. alguns retornam tensor, outros um mapeamento (dict ou
             BatchEncoding -- este último NÃO é subclasse de dict).
        Estratégia: tentar a forma mais rica e degradar em caso de erro.

        `prefixo_forcado` cola um texto DEPOIS do prompt de geração, de modo
        que o modelo continue a partir dele. Usado em dois lugares: no item 15
        (começo neutro de resposta, para deslocar a janela de entropia para
        depois da bifurcação recusar/responder) e no item 14 (cabeçalho do
        canal `final` do gpt-oss, sem o qual o passo lido é marcador de canal
        e não a resposta). Nesse caso a tokenização é feita sobre o texto
        renderizado, porque o template não tem como injetar o sufixo sozinho.
        """
        if prefixo_forcado:
            texto = self._renderizar_texto(mensagens) + prefixo_forcado
            return _normalizar_entrada(
                self.tok(texto, return_tensors="pt", add_special_tokens=False))

        extras = self._extras_template()

        msgs_bloco = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in mensagens
        ]

        tentativas = [
            (msgs_bloco, dict(extras)),
            (mensagens, dict(extras)),
            (msgs_bloco, {}),
            (mensagens, {}),
        ]
        ultimo_erro = None
        for msgs, kw in tentativas:
            try:
                saida = self.template.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    **kw,
                )
                if not kw and extras and not self._aviso_template:
                    print(
                        "  [aviso] o chat template deste modelo ignorou "
                        f"{list(extras)}; verifique se o thinking está de fato desligado."
                    )
                    self._aviso_template = True
                return _normalizar_entrada(saida)
            except Exception as e:  # noqa: BLE001
                ultimo_erro = e

        # último recurso: renderiza o template como texto e tokeniza à mão
        try:
            texto = self.template.apply_chat_template(
                mensagens, add_generation_prompt=True, tokenize=False
            )
            return _normalizar_entrada(
                self.tok(texto, return_tensors="pt", add_special_tokens=False))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"apply_chat_template falhou para {self.hf_id}: {ultimo_erro} / {e}"
            ) from e

    def contar_tokens_prompt(self, mensagens) -> int:
        """
        Conta tokens do prompt SEM gerar nada. Usado pela guarda de VRAM do
        Bloco 1. Não chama limpar_vram: os tensores aqui são de CPU e esta
        função é chamada dezenas de milhares de vezes numa rodada -- rodar o
        gc.collect() completo em cada chamada custaria minutos por nada.
        """
        entrada = self._aplicar_template(mensagens)
        n = int(entrada["input_ids"].shape[-1])
        del entrada
        return n

    # ------------------------------------------------------------------
    # extração da resposta final (remove raciocínio / canal de análise)
    # ------------------------------------------------------------------
    def extrair_resposta_final(self, texto_bruto: str) -> str:
        """
        Devolve só o texto que o usuário veria. Necessário porque, mesmo com
        thinking desligado, o gpt-oss mantém o formato de canais (harmony) e
        alguns templates emitem <think>...</think>.

        IMPORTANTE: o regex de parada precisa rodar sobre ESTE texto, nunca
        sobre o bruto -- um raciocínio que diga "não há informação aqui" antes
        de responder corretamente faria o autômato parar errado.
        """
        t = texto_bruto
        # formato de canais do gpt-oss
        if "<|channel|>final<|message|>" in t:
            t = t.split("<|channel|>final<|message|>")[-1]
        elif "<|message|>" in t:
            t = t.split("<|message|>")[-1]
        # blocos de raciocínio estilo Qwen/Gemma
        for fim in ("</think>", "<end_of_thinking>", "</reasoning>"):
            if fim in t:
                t = t.split(fim)[-1]
        # Tokens de controle residuais.
        #
        # A versão anterior só removia `<|...|>` (com as duas barras) e três
        # tags de raciocínio nomeadas. Isso deixava passar o `</s>` do
        # mistral, que é o EOS dele: `skip_special_tokens=False` é necessário
        # para o alinhamento de canal do gpt-oss, e com ele o EOS de outros
        # modelos sobra no texto. Medido na rodada de 09/09/2026:
        #   mistral, Modalidade A: 93% das `resposta_final` terminavam em
        #     `</s>` -- e é esse texto que alimenta o BERTScore;
        #   mistral, Modalidade B: json_estrito = 0% em TODAS as 2.640
        #     inferências, porque `{"resposta": "..."}</s>` faz o json.loads
        #     falhar com "Extra data". O JSON em si estava perfeito: era
        #     artefato de decodificação, não desobediência do modelo.
        # O César previu exatamente esta classe de falha para o gemma4, cujo
        # template usa `<|turn>` e `<channel|>` -- variantes com UMA barra que
        # o padrão antigo também não pegava. Por isso a limpeza agora cobre as
        # quatro formas e os EOS mais comuns, em vez de nomes específicos.
        t = re.sub(r"<\|[^<>]*?\|>", "", t)      # <|...|>
        t = re.sub(r"<\|[^<>|]*?>", "", t)        # <|...>
        t = re.sub(r"<[^<>|]*?\|>", "", t)        # <...|>
        t = re.sub(r"</?(think|reasoning|analysis|thought|turn|channel|message)>",
                   "", t, flags=re.I)
        t = re.sub(r"</?s>|<pad>|<unk>|\[/?INST\]", "", t)
        return t.strip()

    def _inicio_valor_json(self, ids_gerados, inicio: int = 0):
        """
        Índice do primeiro token do VALOR da chave "resposta".

        Decodifica token a token a partir de `inicio` até casar o padrão
        `"resposta":  "` -- o token seguinte é o começo do texto da resposta.
        Busca limitada a 40 tokens, que é muito mais do que o preâmbulo de um
        objeto JSON de uma chave só.

        Devolve (indice, achou). Quando não acha -- modelo não obedeceu ao
        formato --, devolve o `inicio` original e achou=False, e a posição fica
        marcada no parquet para poder ser filtrada na análise.
        """
        lista = ids_gerados.tolist() if hasattr(ids_gerados, "tolist") else list(ids_gerados)
        padrao = re.compile(r'["\']?' + JSON_CHAVE_RESPOSTA + r'["\']?\s*:\s*["\']')
        acumulado = ""
        for i in range(inicio, min(len(lista), inicio + 40)):
            acumulado += self.tok.decode([lista[i]], skip_special_tokens=False)
            if padrao.search(acumulado):
                return i + 1, True
        return inicio, False

    def _inicio_janela_entropia(self, ids_gerados) -> int:
        """
        Índice do primeiro passo de geração usado no cálculo da entropia.

        Com thinking desligado é 0 (o primeiro token gerado já é resposta),
        que é exatamente o comportamento do experimento.py. Para o gpt-oss,
        pula até depois do último <|message|>, de modo que a entropia meça a
        abertura da RESPOSTA e não a abertura do canal de análise.
        """
        if not self.ids_marcadores:
            return 0
        lista = ids_gerados.tolist() if hasattr(ids_gerados, "tolist") else list(ids_gerados)
        ultimo = -1
        for i, tid in enumerate(lista):
            if tid in self.ids_marcadores:
                ultimo = i
        return ultimo + 1 if ultimo >= 0 else 0

    # ------------------------------------------------------------------
    # geração SEM logits -- Bloco 1
    # ------------------------------------------------------------------
    @torch.no_grad()
    def gerar(self, query, textos_contexto, max_new_tokens: int = MAX_NEW_TOKENS,
              criterio: str = "regex") -> dict:
        """
        Geração enxuta para o Bloco 1.

        `criterio` escolhe o prompt (Modalidade A em prosa ou Modalidade B em
        JSON); a decisão V(y) fica com comum.validar_resposta().

        output_scores=False e return_dict_in_generate=False são OBRIGATÓRIOS
        aqui: no modo incremental o contexto chega a dezenas de páginas e
        materializar um tensor de logits (vocab ~256k floats por passo) para
        cada um dos até 400 passos estouraria a VRAM da 5090 sozinho.
        """
        mensagens = montar_mensagens(query, textos_contexto, criterio=criterio)
        entrada = self._aplicar_template(mensagens)
        entrada = {k: v.to(self.model.device) for k, v in entrada.items()
                   if isinstance(v, torch.Tensor)}
        n_tokens_prompt = int(entrada["input_ids"].shape[-1])

        sequencias = self.model.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=False,               # <-- sem logits neste bloco
            return_dict_in_generate=False,     # <--
            pad_token_id=self.pad_id,
        )
        tokens_gerados = sequencias[0][n_tokens_prompt:]
        n_tokens_resposta = int(len(tokens_gerados))
        texto_bruto = self.tok.decode(tokens_gerados, skip_special_tokens=False)
        resposta = self.extrair_resposta_final(texto_bruto)

        limpar_vram(sequencias, tokens_gerados, entrada)

        return {
            "texto_resposta": resposta,
            "n_tokens_prompt": n_tokens_prompt,
            "n_tokens_resposta": n_tokens_resposta,
            # Item 11: derivável de n_tokens_resposta, mas some na agregação --
            # e importa em dois lugares. Um JSON truncado cai no caminho
            # tolerante de extrair_json_resposta e entra na taxa de
            # conformidade por motivo de COMPRIMENTO; e uma resposta cortada
            # reduz o BERTScore por um motivo que não é ancoragem.
            "atingiu_max_tokens": n_tokens_resposta >= max_new_tokens,
        }

    # ------------------------------------------------------------------
    # geração COM logits + entropia -- Bloco 2
    # ------------------------------------------------------------------
    @torch.no_grad()
    def gerar_com_entropia(
        self,
        query,
        texto_contexto,
        max_new_tokens: int = MAX_NEW_TOKENS,
        n_tokens_entropia: int = N_TOKENS_ENTROPIA,
        prefixo_forcado: str | None = None,
        criterio: str = "regex",
    ) -> dict:
        """
        Geração com captura de logits -- núcleo do Bloco 2.

        Entropia de Shannon EXATA (softmax sobre o vocabulário inteiro, não
        sobre um top-k), na mesma definição do experimento.py.

        MEMÓRIA -- nota corrigida (item 05 da qualificação). `generate()` com
        `return_dict_in_generate=True` RETÉM a pilha inteira de logits até o
        `limpar_vram()` no fim desta função: são ~400 passos x |V| valores, na
        ordem de 200-400 MB por geração, folgados nos 32 GB da 5090. A versão
        anterior deste docstring dizia que a pilha não era guardada, o que era
        falso. A consequência é boa e sustenta os itens 06 a 08: **a pilha já
        está paga**, então medir todos os passos em vez de 20, e extrair
        logprob/margem/top-5 além da entropia, não custa GPU nenhuma. Não
        remova o laço amplo achando que economiza memória -- não economiza.

        LOGITS CRUS (item 01). Usamos `output_logits=True`, não
        `output_scores=True`. `scores` devolve os valores DEPOIS dos logits
        processors: com `do_sample=False` os warpers (temperatura, top-p,
        top-k) são pulados, mas `repetition_penalty`, `no_repeat_ngram_size` e
        `min_new_tokens` continuam entrando. O último é o mais grave aqui:
        joga -inf na posição do EOS exatamente nos primeiros passos, que é
        onde a janela de entropia cai. Ligamos SÓ `output_logits` -- as duas
        flags retêm pilhas independentes e ligar as duas dobraria a memória
        sem ganho.

        INVARIANTE DE ÍNDICE (§G). Com decodificação gulosa, o argmax do logit
        do passo i tem que ser o token gerado no passo i. Verificamos em todos
        os passos (custa ~1 ms) e CONTAMOS as divergências em vez de abortar:
        uma divergência é justamente o sintoma de interferência de processor
        descrita acima, e derrubar uma rodada de 100 h por causa dela seria
        pior do que registrá-la. A taxa vai para metadata_<modelo>.json.

        `prefixo_forcado` (item 15) injeta um começo neutro de resposta antes
        da medição, para deslocar a janela para depois da bifurcação
        recusar/responder.

        `criterio` escolhe a modalidade de parada, igual ao Bloco 1. Sob
        "json", a janela de entropia é deslocada para o primeiro token do
        VALOR da chave (ver ALINHAR_JANELA_ENTROPIA_JSON), senão a medida
        cairia sobre a sintaxe `{"resposta": "` e não sobre a resposta.
        """
        mensagens = montar_mensagens(query, texto_contexto, criterio=criterio)
        entrada = self._aplicar_template(mensagens, prefixo_forcado=prefixo_forcado)
        entrada = {k: v.to(self.model.device) for k, v in entrada.items()
                   if isinstance(v, torch.Tensor)}
        n_tokens_prompt = int(entrada["input_ids"].shape[-1])

        saida = self.model.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_logits=True,                # <-- logit CRU (item 01)
            return_dict_in_generate=True,
            pad_token_id=self.pad_id,
        )

        pilha = saida.logits
        tokens_gerados = saida.sequences[0][n_tokens_prompt:]
        n_tokens_resposta = int(len(tokens_gerados))
        texto_bruto = self.tok.decode(tokens_gerados, skip_special_tokens=False)
        resposta = self.extrair_resposta_final(texto_bruto)
        if prefixo_forcado:
            resposta = (prefixo_forcado + " " + resposta).strip()

        inicio = self._inicio_janela_entropia(tokens_gerados)
        janela_json_alinhada = None
        if criterio == "json" and ALINHAR_JANELA_ENTROPIA_JSON:
            inicio, janela_json_alinhada = self._inicio_valor_json(tokens_gerados, inicio)
        fim = len(pilha) if REGISTRAR_TODOS_OS_PASSOS else min(inicio + n_tokens_entropia,
                                                               len(pilha))
        # guarda: o laço indexa tokens_gerados[i] para o invariante do §G, e
        # em teoria a pilha e a sequência gerada poderiam divergir em tamanho.
        # Truncar aqui evita IndexError e deixa a diferença registrada.
        fim = min(fim, len(tokens_gerados))

        entropias, logprobs_escolhido, p_top1s, margens = [], [], [], []
        top5_tokens, top5_probs = [], []
        n_verificados = n_divergencias = 0

        for i in range(inicio, fim):
            logits = pilha[i][0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            p = logprobs.exp().clamp_min(1e-12)

            # entropia de Shannon exata, em nats
            entropias.append(float(-(p * p.log()).sum().item()))

            # item 07: três escalares do MESMO tensor. A entropia resume a
            # distribuição inteira; a margem e o logprob capturam a decisão
            # local, que é a grandeza natural quando a hipótese é sobre um
            # ponto de bifurcação.
            #
            # Ressalva para o texto da tese: p_top1 e margem são probabilidades
            # em [0,1], mas NÃO são livres da granularidade do tokenizador --
            # um modelo que emite "Não" como um token concentra massa que outro
            # distribui entre "N" + "ão". Comparação entre modelos em nível de
            # escala é sugestiva, não calibrada; o terreno seguro é o posto
            # dentro da pergunta (item 21).
            top2 = torch.topk(p, 2)
            p_top1s.append(float(top2.values[0].item()))
            margens.append(float((top2.values[0] - top2.values[1]).item()))
            tok_gerado = int(tokens_gerados[i].item())
            logprobs_escolhido.append(float(logprobs[tok_gerado].item()))

            # §G: invariante de alinhamento, contado e não asseverado
            n_verificados += 1
            if int(logits.argmax().item()) != tok_gerado:
                n_divergencias += 1

            # item 08: top-5 apenas no PRIMEIRO passo útil. Uma distribuição
            # concentrada não distingue "confiante de que a resposta está
            # aqui" de "confiante de que vai recusar" -- os cinco tokens mais
            # prováveis distinguem, porque um caminho começa em "Não"/"A" e o
            # outro em "O"/"De". É o que permite construir depois uma
            # probabilidade de recusa monotônica, sem a forma em U.
            if i == inicio:
                topk = torch.topk(p, TOP_K_PRIMEIRO_PASSO)
                top5_tokens = [self.tok.decode([int(t)]) for t in topk.indices]
                top5_probs = [float(v) for v in topk.values]

            del logits, logprobs, p, top2

        self.n_passos_verificados += n_verificados
        self.n_divergencias_argmax += n_divergencias

        def _media(v, n=None):
            v = v[:n] if n else v
            return float(np.mean(v)) if v else None

        # item 09: os primeiros tokens decodificados, para inspeção
        # qualitativa do ponto de decisão sem reabrir a GPU.
        prefixo_resposta = self.tok.decode(
            tokens_gerados[inicio:inicio + 10], skip_special_tokens=False)

        limpar_vram(saida, pilha, tokens_gerados, entrada)

        return {
            "texto_resposta": resposta,
            "texto_bruto": texto_bruto,
            "prefixo_resposta": prefixo_resposta,
            # JANELA PRIMÁRIA (20 passos), definida a priori
            "entropia_media": _media(entropias, n_tokens_entropia),
            "logprob_medio": _media(logprobs_escolhido, n_tokens_entropia),
            "p_top1_medio": _media(p_top1s, n_tokens_entropia),
            "margem_media": _media(margens, n_tokens_entropia),
            "n_passos_entropia": min(len(entropias), n_tokens_entropia),
            # REGISTRO COMPLETO (análise secundária pré-declarada)
            "entropia_por_token": entropias,
            "logprob_por_token": logprobs_escolhido,
            "p_top1_por_token": p_top1s,
            "margem_por_token": margens,
            "n_passos_registrados": len(entropias),
            "entropia_media_todos": _media(entropias),
            # item 08
            "top5_tokens_primeiro_passo": top5_tokens,
            "top5_probs_primeiro_passo": top5_probs,
            # itens 03 e §G
            "criterio": criterio,
            "inicio_janela_entropia": int(inicio),
            "janela_json_alinhada": janela_json_alinhada,
            "n_divergencias_argmax": n_divergencias,
            "n_passos_verificados": n_verificados,
            # itens 11 e custo
            "n_tokens_prompt": n_tokens_prompt,
            "n_tokens_resposta": n_tokens_resposta,
            "atingiu_max_tokens": n_tokens_resposta >= max_new_tokens,
        }

    # ------------------------------------------------------------------
    # Item 13 -- teacher forcing do gabarito (um forward, sem decodificação)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def logprob_referencia(self, query, texto_contexto, referencia: str) -> dict:
        """
        Logprob da resposta-gabarito condicionada a (pergunta, chunk).

        Mede diretamente *este chunk sustenta a resposta correta?*. É
        monotônico por construção e não tem a forma em U da entropia, o que o
        torna um comparador limpo ao lado da similaridade de cosseno.

        NORMALIZAÇÃO: devolvemos o logprob MÉDIO POR TOKEN. A soma mediria
        comprimento da referência, não ancoragem.

        DOIS CONFUNDIDORES, e o que fazer com cada um:

        1. Afinidade estilística. O gabarito foi gerado pelo Gemini 1.5 Flash;
           parte do logprob mede quanto o modelo avaliador gosta daquela
           fraseologia. Esse termo é do par (referência, modelo) e NÃO varia
           com o chunk, então entra como deslocamento aproximadamente constante
           nas 10 posições da mesma pergunta e cancela no contraste interno.
           Consequência: a estatística tem que ser o POSTO DENTRO DA PERGUNTA,
           nunca o logprob bruto entre perguntas ou entre modelos.

        2. Cópia lexical -- este NÃO cancela. O gabarito foi gerado lendo o
           chunk-ouro, então costuma citá-lo literalmente. O logprob fica alto
           no gold em parte por sobreposição de superfície, e isso VARIA com o
           chunk, que é justamente o eixo em que o argumento acima se apoia.
           Por isso o 03 grava também `sobreposicao_lexical` (token-F1 entre
           referência e chunk) como covariável: a pergunta defensável passa a
           ser "o posto do gold sob o teacher forcing bate o posto do gold sob
           a mera sobreposição lexical?".
        """
        mensagens = montar_mensagens(query, texto_contexto)
        entrada = self._aplicar_template(mensagens)
        ids_prompt = entrada["input_ids"].to(self.model.device)
        ids_ref = self.tok(referencia, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to(self.model.device)
        n_ref = int(ids_ref.shape[-1])
        if n_ref == 0:
            return {"logprob_medio_referencia": None, "logprob_soma_referencia": None,
                    "n_tokens_referencia": 0}

        ids = torch.cat([ids_prompt, ids_ref], dim=-1)
        saida = self.model(input_ids=ids)
        # logits[t] prevê o token t+1 -> os que preveem a referência começam
        # em (n_prompt - 1) e terminam em (n_total - 2).
        n_prompt = int(ids_prompt.shape[-1])
        logits = saida.logits[0, n_prompt - 1: -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        alvo = ids_ref[0]
        lp = logprobs.gather(-1, alvo.unsqueeze(-1)).squeeze(-1)
        soma = float(lp.sum().item())
        media = float(lp.mean().item())

        limpar_vram(saida, logits, logprobs, lp, ids, ids_prompt, ids_ref)
        return {
            "logprob_medio_referencia": media,
            "logprob_soma_referencia": soma,
            "n_tokens_referencia": n_ref,
        }

    # ------------------------------------------------------------------
    # Item 14 -- sonda sim/não (um forward, um passo, sem decodificação)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sonda_sim_nao(self, query, texto_contexto) -> dict:
        """
        P(sim) para "o documento contém a resposta?". Monotônico, e barato.

        ALINHAMENTO DE CANAL: no gpt-oss o passo 0 é marcador de canal do
        formato harmony, não "Sim"/"Não" -- ler ali mede outra coisa. Em vez
        de gerar livre e procurar onde a resposta começou, forçamos o
        cabeçalho do canal `final` dentro do prompt (`prefixo_canal_final` em
        MODELOS) e lemos o passo seguinte.

        VARIANTES DE SUPERFÍCIE: somamos a massa sobre o PRIMEIRO token de
        cada variante ("sim", "Sim", " Sim", ... e a negativa), em vez de
        apostar em um id único -- "Não" pode sair partido em dois tokens.
        `massa_total` é devolvida: se ficar baixa, o modelo não respondeu no
        formato pedido e o número NÃO é interpretável. Reportar na tabela.

        `p_sim_norm` = massa_sim / (massa_sim + massa_nao) remove o viés de o
        modelo distribuir massa fora do par sim/não.
        """
        pergunta = (
            "Responda apenas com uma palavra: sim ou não.\n\n"
            "INÍCIO DO DOCUMENTO:\n"
            f"{texto_contexto}\n"
            "FIM DO DOCUMENTO\n\n"
            f"O documento acima contém a resposta para a pergunta: \"{query}\"?"
        )
        mensagens = [{"role": "user", "content": pergunta}]
        entrada = self._aplicar_template(
            mensagens, prefixo_forcado=self.cfg.get("prefixo_canal_final"))
        entrada = {k: v.to(self.model.device) for k, v in entrada.items()
                   if isinstance(v, torch.Tensor)}

        saida = self.model(**entrada)
        logits = saida.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        def _massa(variantes):
            ids = set()
            for v in variantes:
                seq = self.tok.encode(v, add_special_tokens=False)
                if seq:
                    ids.add(int(seq[0]))          # primeiro token da variante
                tid = self.tok.convert_tokens_to_ids(v)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(tid)
            return float(sum(float(probs[i].item()) for i in ids if i < probs.shape[0]))

        massa_sim = _massa(SONDA_VARIANTES_SIM)
        massa_nao = _massa(SONDA_VARIANTES_NAO)
        total = massa_sim + massa_nao

        limpar_vram(saida, logits, probs, entrada)
        return {
            "sonda_massa_sim": massa_sim,
            "sonda_massa_nao": massa_nao,
            "sonda_massa_total": total,
            "sonda_p_sim_norm": (massa_sim / total) if total > 1e-9 else None,
        }

    # ------------------------------------------------------------------
    # Item 12 -- registro de ambiente
    # ------------------------------------------------------------------
    def metadata(self, extra: dict | None = None) -> dict:
        """
        Tudo o que a tese precisa citar sobre COMO este número foi produzido.
        Escrito automaticamente ao fim de cada rodada, para não depender de um
        `pip freeze` avulso no final.
        """
        import platform
        import transformers

        def _v(mod):
            try:
                return __import__(mod).__version__
            except Exception:  # noqa: BLE001
                return None

        meta = {
            "tag_modelo": self.tag,
            "hf_id": self.hf_id,
            "quantizacao": self.quant,
            # mesmos nomes usados na nota de ajustes de 04/09/2026
            "attn_implementation_pedida": self.attn_implementation_pedida,
            "attn_implementation_efetiva": self.attn_implementation_efetiva,
            "vocab_size": self.vocab_size,          # item 02 (reduzido)
            "thinking_desligado": self.cfg.get("thinking") is False,
            "reasoning_effort": self.cfg.get("reasoning_effort"),
            "marcadores_fim_prefacio": self.cfg.get("marcadores_fim_prefacio", []),
            "ids_marcadores_resolvidos": self.ids_marcadores,
            "prefixo_canal_final": self.cfg.get("prefixo_canal_final"),
            "limite_tokens_prompt": self.limite_tokens_prompt,
            "generation_config_efetivo": self.generation_config_efetivo,
            # §G: invariante de índice de logit, contado ao longo da rodada
            "n_passos_verificados": self.n_passos_verificados,
            "n_divergencias_argmax": self.n_divergencias_argmax,
            "taxa_divergencia_argmax": (
                self.n_divergencias_argmax / self.n_passos_verificados
                if self.n_passos_verificados else None),
            "seed_base": SEED_BASE,
            "n_tokens_entropia_primaria": N_TOKENS_ENTROPIA,
            "registrar_todos_os_passos": REGISTRAR_TODOS_OS_PASSOS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "versoes": {
                "python": platform.python_version(),
                "torch": _v("torch"),
                "transformers": transformers.__version__,
                "bitsandbytes": _v("bitsandbytes"),
                "accelerate": _v("accelerate"),
                "numpy": _v("numpy"),
                "pandas": _v("pandas"),
            },
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "vram_pico_gb": vram_pico_gb(),
        }
        meta.update(extra or {})
        return meta

    # ------------------------------------------------------------------
    def descarregar(self):
        """
        Libera o modelo da VRAM. Chamar antes de carregar o próximo.

        `self.model = None` fica no FINALLY de propósito. Antes ele era a
        última linha do corpo: se `limpar_vram()` levantasse -- e ela chama
        `torch.cuda.ipc_collect()`, que pode falhar dentro de contêiner --, a
        exceção subia, o atributo continuava apontando para o modelo e a VRAM
        NÃO era liberada. Como os scripts envolvem esta chamada num
        `except: pass`, a falha era invisível. Rodando um modelo por processo
        isso não aparece; rodando os seis num processo só (o comando sem
        `--modelos`), o segundo modelo entra numa placa que ainda tem o
        primeiro inteiro dentro.
        """
        try:
            try:
                self.model.to("meta")
            except Exception:
                pass
            limpar_vram(self.model, self.proc, self.tok)
        except Exception as e:  # noqa: BLE001
            # visível, não engolida: se a limpeza falha, o próximo modelo herda
            # a VRAM ocupada e o erro só aparece como OOM inexplicável
            print(f"  [ATENÇÃO] falha ao liberar a VRAM ({type(e).__name__}: {e}). "
                  f"Se for rodar outro modelo neste mesmo processo, prefira um "
                  f"processo por modelo.")
        finally:
            self.model = None


# ============================================================
# 7b. SOBREPOSIÇÃO LEXICAL -- covariável de controle do item 13
# ============================================================

def sobreposicao_lexical(referencia: str, chunk: str) -> float:
    """
    Token-F1 entre a resposta de referência e o texto do chunk.

    Controle para o confundidor de cópia lexical do item 13: o gabarito das 74
    foi gerado por um LLM que estava LENDO o chunk-ouro, então frequentemente
    o cita. Sem esta covariável, "o teacher forcing acha o gold" pode ser só
    "o gabarito copia o gold". Custa nada e roda na CPU.
    """
    a = re.findall(r"\w+", str(referencia).lower())
    b = set(re.findall(r"\w+", str(chunk).lower()))
    if not a or not b:
        return 0.0
    acertos = sum(1 for t in a if t in b)
    prec = acertos / len(a)
    rec = acertos / len(b)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


# ============================================================
# 7c. DIAGNÓSTICO DE CIRCULARIDADE DA ENTROPIA
# ============================================================

# A entropia é um substituto da decisão de parada quando o AUC se afasta de
# 0,5 para QUALQUER um dos lados. 0,5 = nenhuma informação.
#
# Os dois extremos são circulares, e o perigoso é o de baixo:
#   AUC -> 1  as recusas têm a menor entropia. O argmin é sempre uma recusa e
#             "menor entropia é o gold" cai a ~0 por construção. Barulhento,
#             fácil de perceber.
#   AUC -> 0  as válidas têm a menor entropia. O argmin é sempre uma posição
#             que respondeu -- e quando o gold é o único chunk que sustenta
#             resposta, ele é apontado por construção. A hipótese da tese sai
#             CONFIRMADA sem que a entropia tenha dito nada sobre relevância.
# Por isso o alerta é bilateral.
AUC_CIRCULARIDADE_ALERTA = 0.95


def diagnostico_circularidade(entropias, validas) -> dict:
    """
    A entropia está medindo relevância, ou está medindo "o modelo recusou"?

    Por que isto precisa ser automático. A hipótese do Bloco 2 é que a
    entropia baixa aponta o chunk que sustenta a resposta. Mas uma recusa
    também pode ter entropia baixa -- e aí o argmin aponta a recusa mais
    confiante, não a resposta. O README já pedia para olhar a tabela
    "válida x recusa"; com 74 perguntas x 4 modelos x 2 modalidades ninguém
    olha. Então aqui vira número, com limiar declarado.

    O caso severo é a SEPARAÇÃO TOTAL: quando a maior entropia entre as
    recusas é menor que a menor entropia entre as respostas válidas, o
    `argmin` é uma função determinística de `valida`. A entropia não
    acrescenta nada à decisão de parada -- ela É a decisão de parada, escrita
    em nats. Foi exatamente o que o piloto de 05/09/2026 encontrou na
    Modalidade B: a sentinela "NÃO ENCONTRADO" é uma string fixa, e emitir
    uma string fixa custa ~1e-4 nats por token.

    Devolve AUC (probabilidade de uma recusa sorteada ter entropia menor que
    uma válida sorteada), os extremos das duas distribuições e as bandeiras.
    """
    v = np.asarray([x for x in entropias], dtype=float)
    ok = np.asarray([bool(x) for x in validas], dtype=bool)
    m = ~np.isnan(v)
    v, ok = v[m], ok[m]
    val, rec = v[ok], v[~ok]
    if not len(val) or not len(rec):
        return {"auc": None, "n_validas": int(len(val)), "n_recusas": int(len(rec)),
                "teto_recusa": None, "piso_valida": None,
                "separacao_total": None, "circular": None}
    d = np.subtract.outer(rec, val)
    auc = float(((d < 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    sep_recusa_menor = bool(rec.max() < val.min())    # recusas todas abaixo
    sep_valida_menor = bool(val.max() < rec.min())    # válidas todas abaixo
    return {
        "auc": auc,
        "n_validas": int(len(val)),
        "n_recusas": int(len(rec)),
        "teto_recusa": float(rec.max()),
        "piso_recusa": float(rec.min()),
        "teto_valida": float(val.max()),
        "piso_valida": float(val.min()),
        "separacao_total": sep_recusa_menor or sep_valida_menor,
        "sentido": ("recusa_menor" if auc > 0.5 else
                    "valida_menor" if auc < 0.5 else "empate"),
        "circular": bool(auc >= AUC_CIRCULARIDADE_ALERTA
                         or auc <= 1 - AUC_CIRCULARIDADE_ALERTA),
    }


def texto_alerta_circularidade(d: dict, rotulo: str = "") -> list[str]:
    """Linhas de aviso prontas para imprimir. Vazio quando está tudo bem."""
    if not d or d.get("auc") is None or not d.get("circular"):
        return []
    pre = f"[{rotulo}] " if rotulo else ""
    recusa_menor = d["auc"] > 0.5
    linhas = [
        f"  <<< ATENÇÃO {pre}a entropia é quase um substituto da decisão de "
        f"parada (AUC={d['auc']:.3f}).",
    ]
    if d["separacao_total"]:
        if recusa_menor:
            linhas.append(f"      SEPARAÇÃO TOTAL: recusa <= "
                          f"{d['teto_recusa']:.6f} < {d['piso_valida']:.6f} <= válida.")
        else:
            linhas.append(f"      SEPARAÇÃO TOTAL: válida <= "
                          f"{d['teto_valida']:.6f} < {d['piso_recusa']:.6f} <= recusa.")
    if recusa_menor:
        linhas += [
            "      argmin(entropia) é sempre uma RECUSA, então 'menor entropia "
            "é o gold' cai a",
            "      ~0 por construção -- não por ausência de sinal de relevância.",
        ]
    else:
        linhas += [
            "      argmin(entropia) é sempre uma posição que RESPONDEU. Este é "
            "o sentido",
            "      PERIGOSO: quando o gold é o único chunk que sustenta "
            "resposta, ele é",
            "      apontado por construção e a hipótese sai confirmada sem que "
            "a entropia",
            "      tenha dito nada sobre relevância. Compare contra V (nº de "
            "posições",
            "      válidas): se V=1 na maioria das perguntas, o resultado é "
            "tautológico.",
        ]
    linhas += [
        "      Nos dois casos a entropia não é um sinal independente. Reporte-a "
        "como medida",
        "      de conformidade/decisão, não de ancoragem, ou restrinja a "
        "comparação a um",
        "      subconjunto homogêneo (e diga que restringiu).",
    ]
    return linhas


# ============================================================
# 7d. NULA ESTRATIFICADA POR VALIDADE (permutação)
# ============================================================

N_PERM_NULA = 20000


def amostrar_nula_estratificada(grupos, sinal: int = -1, estratificar: bool = True,
                                n_perm: int = N_PERM_NULA, semente: int = 20260905):
    """
    Nula de "o escore não sabe nada sobre o gold ALÉM do que a validade já sabe".

    O problema que isto resolve. A nula ingênua do item 18 permuta as 10
    posições livremente, o que supõe que a entropia é permutável entre elas.
    Ela não é: recusa e resposta válida têm distribuições de entropia
    diferentes (é o que o AUC de 7c mede). Com isso a nula ingênua fica
    ERRADA nos dois sentidos, e o sentido do erro depende do sinal do
    acoplamento -- ora acusa sinal onde só há estrutura de validade, ora
    esconde sinal real atrás dela.

    A correção é permutar DENTRO de cada estrato de validade. Isso preserva
    exatamente o acoplamento escore-validade observado e pergunta só o que
    interessa: dentro do grupo a que o gold pertence, ele se destaca?

    Por que isto é melhor do que restringir a análise às posições válidas:
    mantém as 10 posições e todas as perguntas (inclusive aquelas em que o
    gold NÃO é válido, que são justamente as falhas informativas), e não
    precisa de nula analítica -- a permutação já a produz.

    `grupos`: lista com uma tupla (escores, is_gold, estrato) por pergunta.
    `sinal` : -1 quando MENOR é melhor (entropia); +1 quando maior é melhor.
    Devolve (postos, argmin) com forma (n_perm, n_perguntas).

    ATALHO: não é preciso permutar as 10 posições. O multiconjunto de escores
    da pergunta é fixo, então o posto do gold depende SÓ de qual valor o gold
    recebe -- e permutar dentro do estrato é o mesmo que sortear, sem
    reposição, quais casas daquele estrato ficam com os golds. Basta então
    pré-calcular o posto de cada casa e sortear casas. Isso derruba o custo de
    O(n_perm . k . log k) para O(n_perm) na esmagadora maioria das perguntas
    (as de gold único), e é o que torna a rodada completa viável.
    """
    rng = np.random.default_rng(semente)
    postos = np.empty((n_perm, len(grupos)), dtype=float)

    for j, (escores, is_gold, estrato) in enumerate(grupos):
        v = np.asarray(escores, dtype=float)
        g = np.asarray(is_gold, dtype=bool)
        s = np.asarray(estrato, dtype=bool)
        k = len(v)
        chave = -sinal * v                      # ordenar ascendente = melhor primeiro

        # posto de cada CASA, fixo: o conjunto de valores não muda
        ordem = np.argsort(chave, kind="stable")
        posto_casa = np.empty(k, dtype=np.int64)
        posto_casa[ordem] = np.arange(1, k + 1)

        blocos = [s, ~s] if estratificar else [np.ones(k, dtype=bool)]
        melhor = np.full(n_perm, k + 1, dtype=np.int64)
        for m in blocos:
            casas = np.where(m)[0]
            n_g = int(g[m].sum())               # golds neste estrato
            if n_g == 0 or len(casas) == 0:
                continue
            pc = posto_casa[casas]
            if n_g == 1:
                # caso dominante (62 das 74 perguntas): um sorteio uniforme
                sorteio = pc[rng.integers(0, len(casas), size=n_perm)]
            else:
                # multi-gold: n_g casas distintas do estrato, melhor posto
                idx = np.argsort(rng.random((n_perm, len(casas))), axis=1)[:, :n_g]
                sorteio = pc[idx].min(axis=1)
            melhor = np.minimum(melhor, sorteio)
        postos[:, j] = melhor

    return postos, (postos == 1)


def p_valor_nula(observado: float, amostras: np.ndarray, menor_e_melhor: bool) -> float:
    """p unilateral por permutação, com a correção +1 (Phipson & Smyth)."""
    a = np.asarray(amostras, dtype=float)
    extremos = (a <= observado) if menor_e_melhor else (a >= observado)
    return float((int(extremos.sum()) + 1) / (len(a) + 1))


def nula_restrita_exata(V, gold_valido) -> float:
    """
    Nula do "menor escore entre as VÁLIDAS é o gold", quando se opta por
    restringir a análise ao subconjunto válido.

    E[nula] = média_i( 1{gold_i válido} / V_i ).

    ARMADILHA DE JENSEN, que já custou uma leitura errada nesta rodada:
    isto NÃO é 1/média(V). Como 1/x é convexa, média(1/V) > 1/média(V), e a
    diferença é grande justamente quando alguma pergunta tem V pequeno --
    com V=1 a estatística restrita vale 1 por construção, não por sinal.
    No piloto de 05/09/2026: 1/média(V) = 0,208 contra média(1/V) = 0,360.
    """
    v = np.asarray(V, dtype=float)
    ok = np.asarray(gold_valido, dtype=bool)
    m = v > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.where(ok[m], 1.0 / v[m], 0.0)))


# ============================================================
# 8. BERTSCORE
# ============================================================

def _patch_tokenizer_model_max_length():
    """
    Corrige o bug do bert_score com transformers recentes: tokenizers sem
    `model_max_length` no config (caso do neuralmind/bert-base-portuguese-cased)
    recebem ~1e30 como padrão, e o backend Rust estoura ao converter para int
    ("OverflowError: int too big to convert"). Força 512, o padrão do BERT.
    Idêntico ao patch do experimento.py.
    """
    from transformers import AutoTokenizer

    if getattr(AutoTokenizer.from_pretrained, "_patched_max_length", False):
        return
    original = AutoTokenizer.from_pretrained

    def patched(*args, **kwargs):
        tok = original(*args, **kwargs)
        if getattr(tok, "model_max_length", None) and tok.model_max_length > 100_000:
            tok.model_max_length = 512
        return tok

    patched._patched_max_length = True
    AutoTokenizer.from_pretrained = patched


def calcular_bertscore(geradas, referencias):
    """Retorna (P, R, F1) como listas de float. Roda em CPU se não houver GPU livre."""
    from bert_score import score as bertscore_score

    _patch_tokenizer_model_max_length()
    geradas = [str(x) if x is not None else "" for x in geradas]
    referencias = [str(x) if x is not None else "" for x in referencias]
    P, R, F1 = bertscore_score(
        geradas,
        referencias,
        lang="pt",
        model_type=BERTSCORE_MODEL_TYPE,
        num_layers=BERTSCORE_NUM_LAYERS,
    )
    return P.tolist(), R.tolist(), F1.tolist()
