"""
verificar_memoria.py -- sonda de memória ANTES de liberar horas de GPU.

O `verificar_ambiente.py` confirma que a placa funciona. Este confirma que
UM modelo específico cabe nela no pior caso do desenho: o braço `fixo50`,
que injeta as 50 páginas do topo de uma vez e chega perto do teto de
`limite_tokens_prompt`.

Por que isso não é paranoia: a memória da atenção depende da implementação
que o transformers escolhe para cada arquitetura. Quando ela cai para
"eager", o custo é QUADRÁTICO no comprimento do prompt -- num modelo de 64
cabeças a 32k tokens são 137 GB só para a matriz de scores, e a rodada
morre na primeira pergunta. Com SDPA ou flex_attention o custo é linear.
Três minutos aqui evitam descobrir isso na pergunta 1 de 192.

Uso:
  python verificar_memoria.py --modelos gemma3:27b
  python verificar_memoria.py                       # todos os configurados

Saída: pico de VRAM e tempo de prefill para prompts curto e longo, mais a
implementação de atenção efetivamente usada. Sai com código 1 se algum
modelo falhar no prompt longo.
"""

import argparse
import time
import traceback

import torch

import comum

# comprimentos sondados: o curto e o pior caso realista do fixo50
COMPRIMENTOS = (2048, 30000)
MARGEM_ALERTA_GB = 3.0     # pico a menos disto do total ja e zona de risco


def sondar(tag: str, attn: str | None = None,
           quant: str | None = None) -> bool:
    print("=" * 70)
    print(f"{tag}")
    print("=" * 70)
    ok = True
    gerador = None
    try:
        gerador = comum.Gerador(tag, verboso=True, attn=attn, quant=quant)
        cfg_attn = getattr(gerador.model.config, "_attn_implementation", "?")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"  atencao efetiva: {cfg_attn}")
        print(f"  limite_tokens_prompt: {gerador.limite_tokens_prompt}")

        for n in COMPRIMENTOS:
            if n > gerador.limite_tokens_prompt:
                print(f"  {n:>6} tok: pulado (acima do limite configurado)")
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                ids = torch.randint(100, min(gerador.vocab_size, 100000),
                                    (1, n)).to(gerador.model.device)
                t0 = time.time()
                with torch.no_grad():
                    gerador.model.generate(
                        input_ids=ids, max_new_tokens=8, do_sample=False,
                        pad_token_id=gerador.pad_id)
                dt = time.time() - t0
                pico = torch.cuda.max_memory_allocated() / 2**30
                folga = total_gb - pico
                if folga < 0:
                    marca = "[NAO CABE]"
                elif folga < MARGEM_ALERTA_GB:
                    marca = "[RISCO]"
                else:
                    marca = "[ok]  "
                print(f"  {marca} {n:>6} tok: {dt:6.1f}s | pico {pico:5.1f} GB "
                      f"| folga {folga:5.1f} GB de {total_gb:.1f} GB")
                if folga < 0:
                    # Pico acima da VRAM total: no Windows/WSL o driver NVIDIA
                    # NAO da OOM -- ele pagina para a memoria compartilhada.
                    # A rodada nao quebra, ela fica ordens de grandeza mais
                    # lenta, o que e pior porque passa despercebido.
                    ok = False
                    print("          NAO CABE: o pico excede a VRAM total em "
                          f"{-folga:.1f} GB. O driver vai paginar para a RAM e "
                          "a rodada vai arrastar.")
                elif folga < MARGEM_ALERTA_GB:
                    print("          margem apertada: uma pergunta com paginas "
                          "mais longas pode estourar.")
                del ids
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"  [FALHA] {n:>6} tok: {type(e).__name__}: {str(e)[:200]}")
    except Exception:  # noqa: BLE001
        ok = False
        print("  [FALHA] no carregamento:")
        traceback.print_exc()
    finally:
        if gerador is not None:
            del gerador
        comum.limpar_vram()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default=None,
                    help="tags separadas por virgula; padrao: todas")
    ap.add_argument("--attn", default=None,
                    help="sobrescreve attn_implementation (ex.: flex_attention, "
                         "sdpa, eager). Serve para comparar backends antes de "
                         "queimar horas de GPU numa rodada que pagina VRAM")
    ap.add_argument("--quant", default=None, choices=["4bit", "8bit", "nativo"],
                    help="sobrescreve a quantizacao do MODELOS")
    args = ap.parse_args()
    tags = ([t.strip() for t in args.modelos.split(",") if t.strip()]
            if args.modelos else list(comum.MODELOS))

    resultados = {tag: sondar(tag, attn=args.attn, quant=args.quant)
                  for tag in tags}

    print()
    print("=" * 70)
    for tag, ok in resultados.items():
        print(f"  {'[ok]   ' if ok else '[FALHA]'} {tag}")
    if all(resultados.values()):
        print("Memoria OK. Pode liberar a maquina para estes modelos.")
        raise SystemExit(0)
    print("Ha modelo que NAO cabe no pior caso. Nao inicie a rodada dele.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
