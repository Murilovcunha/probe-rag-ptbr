"""
verificar_ambiente.py -- Checagem de sanidade ANTES de gastar horas de GPU.

Confirma, em ordem:
  1. versões de torch / transformers / bitsandbytes / accelerate;
  2. que a GPU é vista pelo torch e que a arquitetura da placa é suportada
     pelo build instalado (a RTX 5090 é Blackwell, sm_120: builds de torch
     compilados só até sm_90 enxergam a placa mas falham no primeiro kernel,
     com "no kernel image is available for execution on the device");
  3. que uma multiplicação de matrizes de verdade roda na GPU;
  4. que o bitsandbytes consegue quantizar em 4-bit nesta placa;
  5. que os arquivos de dados esperados existem e têm o formato certo;
  6. quanto de VRAM está livre.

Uso:
  python verificar_ambiente.py
"""

import importlib
import os
import sys

import numpy as np

OK, FALHA, AVISO = "[ok]   ", "[FALHA]", "[aviso]"
problemas = []


def versao(nome):
    try:
        m = importlib.import_module(nome)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"AUSENTE ({type(e).__name__})"


def main():
    print("=" * 70)
    print("1. Pacotes")
    print("=" * 70)
    for pacote in ("torch", "transformers", "accelerate", "bitsandbytes",
                   "sentence_transformers", "pandas", "pyarrow", "bert_score"):
        v = versao(pacote)
        marca = FALHA if "AUSENTE" in str(v) else OK
        if marca == FALHA:
            problemas.append(f"pacote ausente: {pacote}")
        print(f"{marca} {pacote:<24} {v}")

    print()
    print("=" * 70)
    print("2. GPU")
    print("=" * 70)
    try:
        import torch
    except Exception:
        print(f"{FALHA} torch não importa. Nada mais a testar.")
        sys.exit(1)

    print(f"       torch.version.cuda        = {torch.version.cuda}")
    print(f"       torch.cuda.is_available() = {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        problemas.append("torch não vê nenhuma GPU")
        print(f"{FALHA} sem GPU visível.")
    else:
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}"
            total = props.total_memory / 1024 ** 3
            print(f"{OK} GPU {i}: {props.name}  |  compute capability {cc}  |  {total:.1f} GB")
        arqs = torch.cuda.get_arch_list()
        print(f"       arquiteturas no build     = {arqs}")
        props = torch.cuda.get_device_properties(0)
        alvo = f"sm_{props.major}{props.minor}"
        if alvo not in arqs:
            problemas.append(
                f"o build do torch não inclui {alvo}: reinstale com o índice cu128 ou superior")
            print(f"{FALHA} {alvo} NÃO está no build. Kernels vão falhar em tempo de execução.")
            print("        Corrija com:")
            print("        pip install --force-reinstall torch torchvision "
                  "--index-url https://download.pytorch.org/whl/cu128")
        else:
            print(f"{OK} {alvo} presente no build.")

        print()
        print("=" * 70)
        print("3. Kernel de verdade na GPU")
        print("=" * 70)
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            b = a @ a
            torch.cuda.synchronize()
            print(f"{OK} matmul bfloat16 2048x2048 executou (soma={float(b.float().sum()):.1f})")
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            problemas.append(f"kernel CUDA falhou: {e}")
            print(f"{FALHA} {type(e).__name__}: {e}")

        print()
        print("=" * 70)
        print("4. bitsandbytes em 4-bit nesta placa")
        print("=" * 70)
        try:
            import bitsandbytes as bnb

            camada = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16).cuda()
            x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
            y = camada(x)
            torch.cuda.synchronize()
            print(f"{OK} Linear4bit rodou, saída {tuple(y.shape)}")
            del camada, x, y
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            problemas.append(f"bitsandbytes 4-bit falhou: {e}")
            print(f"{FALHA} {type(e).__name__}: {e}")
            print("        Tente: pip install -U bitsandbytes")

        livre, total = torch.cuda.mem_get_info()
        print()
        print(f"       VRAM livre agora: {livre / 1024 ** 3:.1f} GB de "
              f"{total / 1024 ** 3:.1f} GB")
        if livre / 1024 ** 3 < 24:
            print(f"{AVISO} menos de 24 GB livres. Feche o que estiver usando a GPU "
                  f"(navegador, Ollama, Jupyter) antes de rodar os experimentos.")

    print()
    print("=" * 70)
    print("5. Dados")
    print("=" * 70)
    import comum

    esperados = {
        comum.ARQ_CORPUS: ["id", "texto_artigo", "texto_pagina", "vector"],
        comum.ARQ_PERGUNTAS_192: ["pergunta_idx", "pergunta", "resposta_gabarito"],
        comum.ARQ_PERGUNTAS_74: ["pergunta_idx", "pergunta", "id_chunk_gold",
                                 "ids_chunk_gold_todos"],
    }
    import pandas as pd

    for caminho, colunas in esperados.items():
        if not os.path.exists(caminho):
            problemas.append(f"arquivo ausente: {caminho}")
            print(f"{FALHA} {caminho} não existe (rode 00_preparar_dados.py)")
            continue
        df = pd.read_parquet(caminho)
        faltando = [c for c in colunas if c not in df.columns]
        if faltando:
            problemas.append(f"{caminho}: colunas ausentes {faltando}")
            print(f"{FALHA} {caminho}: faltam {faltando}")
        else:
            print(f"{OK} {caminho}  ({len(df)} linhas)")

    for caminho in (comum.ARQ_EMB_192, comum.ARQ_EMB_74):
        if os.path.exists(caminho):
            df = pd.read_parquet(caminho)
            dim = len(df["embedding_pergunta"].iloc[0])
            print(f"{OK} {caminho}  ({len(df)} linhas, dim={dim})")
        else:
            print(f"{AVISO} {caminho} ainda não existe "
                  f"(rode 01_embeddings_perguntas.py)")

    # consistência ranking: o embedding do corpus deve estar normalizado
    if os.path.exists(comum.ARQ_CORPUS):
        c = pd.read_parquet(comum.ARQ_CORPUS)
        v = np.asarray(c["vector"].iloc[0], dtype=np.float32)
        print(f"       dim do embedding do corpus = {len(v)}, "
              f"norma = {np.linalg.norm(v):.4f}")

    print()
    print("=" * 70)
    print("6. Normalização do chat template (regressão do BatchEncoding)")
    print("=" * 70)
    # O transformers devolve BatchEncoding, que herda de UserDict e NÃO é
    # subclasse de dict. Um teste isinstance(x, dict) embrulha o objeto inteiro
    # e quebra tudo com KeyError: 'shape'. Custa nada checar aqui, offline,
    # antes de baixar 13 GB de pesos para descobrir isso no piloto.
    from collections import UserDict

    class _BE(UserDict):
        pass

    try:
        import torch as _t
        casos = {
            "dict": {"input_ids": _t.tensor([[1, 2, 3]])},
            "BatchEncoding": _BE({"input_ids": _t.tensor([[1, 2, 3]]),
                                  "attention_mask": _t.tensor([[1, 1, 1]])}),
            "tensor cru": _t.tensor([[1, 2, 3]]),
        }
        for nome, v in casos.items():
            r = comum._normalizar_entrada(v)
            assert isinstance(r, dict) and hasattr(r["input_ids"], "shape")
            print(f"{OK} {nome:<14} -> dict com input_ids.shape="
                  f"{tuple(r['input_ids'].shape)}")
    except Exception as e:  # noqa: BLE001
        problemas.append(f"normalização do chat template falhou: {e}")
        print(f"{FALHA} {type(e).__name__}: {e}")

    print()
    print("=" * 70)
    if problemas:
        print(f"{len(problemas)} problema(s) a resolver antes de rodar:")
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)
    print("Ambiente OK. Pode rodar os experimentos.")


if __name__ == "__main__":
    main()
