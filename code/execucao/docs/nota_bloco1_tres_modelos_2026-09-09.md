# Bloco 1 — três modelos sob `e5small`: o efeito de +0,02 não replica

Nota da consolidação de 09/09/2026, com `gpt-oss:20b`, `mistral-small3.1` e
`qwen3.8:27b` concluídos. Complementa `nota_rodada_gptoss_2026-09-07.md`, que
cobre a rodada isolada do `gpt-oss` e a análise do teto de tokens.

| | |
|---|---|
| commit | `0314aac` |
| retriever | `e5small` · `D_max` 50 · `MAX_NEW_TOKENS` 800 |
| escopo | 3 modelos x 192 perguntas x 2 critérios x 3 modos = 3.456 varreduras |
| tempos | `gpt-oss` ~7 h · `mistral` ~9,5 h · `qwen3.8` ~31 h |
| backends | `gpt-oss` flex_attention · `mistral` sdpa · `qwen3.8` sdpa (obrigatório) |

---

## 1. A pergunta que esta rodada tinha de responder

Da rodada isolada do `gpt-oss` saiu um `iterativo − fixo50` de **+0,020** em
BERTScore F1 sob o critério regex. A orientação foi: *"um efeito de +0,02 que
se repita em vários modelos vira evidência; num só, não vira"*.

Ficou registrada, antes desta rodada, a ressalva de que a replicação **não**
decidiria a questão sozinha: o `fixo50` responder mais longo que o `iterativo`
é estrutural (50 chunks de contexto contra 1), não é característica de um
modelo, e portanto se repetiria nos seis independentemente de haver diferença
de qualidade. O que decidiria seria o **recall** e a **distribuição de
comprimento**, ambos já gravados em toda rodada.

## 2. Não replica

`iterativo − fixo50`, pareado por pergunta, n = 192 em cada célula:

| modelo | critério | ΔF1 | ΔR |
|---|---|---|---|
| gpt-oss:20b | regex | **+0,0201** | −0,0134 |
| qwen3.8:27b | regex | +0,0005 | −0,0330 |
| mistral-small3.1 | regex | −0,0081 | −0,0318 |
| gpt-oss:20b | json | −0,0070 | −0,0397 |
| qwen3.8:27b | json | −0,0283 | −0,0505 |
| mistral-small3.1 | json | −0,0330 | −0,0505 |

Uma célula positiva em seis, e é aquela em que a hipótese nasceu. As outras
cinco estão em zero ou negativas.

**O recall é negativo nas seis células**, de −0,013 a −0,051. O baseline de 50
chunks recupera mais conteúdo do gabarito em todos os modelos e nas duas
modalidades, sem exceção.

## 3. O mecanismo, confirmado nos três modelos

`S_bertscore_f1` é não-monotônico no comprimento da resposta: máximo na faixa
de razão 0,5–1 (0,800), caindo para 0,67 em 2–4 e 0,53 acima de 8. A
decomposição mostra por quê — o recall se sustenta e a **precisão desaba**
(0,82 → 0,45). O F1 pune verbosidade, não erro.

Razão mediana comprimento-resposta / comprimento-gabarito:

| modelo | critério | fixo50 | incremental | iterativo |
|---|---|---|---|---|
| gpt-oss:20b | regex | 1,92 | 1,25 | 1,17 |
| qwen3.8:27b | regex | 1,89 | 1,33 | 1,26 |
| mistral-small3.1 | regex | 1,39 | 1,20 | 1,12 |
| gpt-oss:20b | json | 1,09 | 0,92 | 0,81 |
| qwen3.8:27b | json | 0,99 | 0,91 | 0,85 |
| mistral-small3.1 | json | 1,00 | 0,80 | 0,74 |

O sinal do ΔF1 é previsto pela posição de cada braço na curva:

- sob **regex**, o `fixo50` está em 1,4–1,9 (passado do ótimo) e o `iterativo`
  em 1,1–1,3 (mais perto) → ΔF1 tende a positivo;
- sob **json**, o `fixo50` cai em ~1,0 (em cima do ótimo) e o `iterativo` em
  0,74–0,85 (curto demais) → ΔF1 inverte e fica negativo.

Nos três modelos, nas duas modalidades. Não sobra efeito de método a explicar.

Na rodada do `gpt-oss` isso já tinha sido isolado por outro caminho: dentro de
cada faixa de comprimento os três braços são indistinguíveis, e o que difere é
a **distribuição** (46 % das respostas do `fixo50` com razão > 2, contra 23 %
do `iterativo`).

## 4. O que replica, e é o resultado

**η do `iterativo` é positivo nos três modelos:**

| modelo | η regex | η json |
|---|---|---|
| gpt-oss:20b | 0,925 | 0,814 |
| qwen3.8:27b | 0,760 | 0,649 |
| mistral-small3.1 | 0,715 | 0,615 |

**η do `incremental` vira negativo em dois dos três:**

| modelo | η regex | η json | mediana |
|---|---|---|---|
| gpt-oss:20b | 0,749 | 0,336 | 0,96 / 0,96 |
| mistral-small3.1 | **−0,280** | **−0,710** | 0,94 / 0,94 |
| qwen3.8:27b | **−0,732** | **−1,457** | 0,96 / 0,95 |

As medianas continuam em ~0,95: a média negativa é inteiramente cauda. Em dois
de três modelos, **a construção incremental gasta em média mais tokens do que
ler os 50 chunks de uma vez**. `censurado_limite_tokens` = 2,5 % no incremental
(11 e 7 casos no qwen, 7 e 4 no mistral).

**Leitura que os dados sustentam:** o método iterativo entrega recall de 1,3 a
5,1 pontos menor que o baseline de 50 chunks, a ~7–24 % do custo em tokens. É
um trade-off quantificado, replicado em três modelos. O incremental não tem o
lado do custo a seu favor e por isso não sustenta a mesma frase.

## 5. A dificuldade das 192 perguntas é em blocos, e a censura se concentra

`n_inferencias` média por quartil de `pergunta_idx` (braços com parada):

| modelo | 0–47 | 48–95 | 96–143 | 144–191 |
|---|---|---|---|---|
| gpt-oss:20b | 3,38 | 1,18 | 2,98 | 3,22 |
| mistral-small3.1 | 4,43 | 2,72 | 7,10 | 8,43 |
| qwen3.8:27b | 4,97 | 1,77 | 7,59 | **10,59** |

`D_iter` médio nas mesmas faixas é muito mais plano (2–4 em quase todas). A
diferença está em `n_inferencias`, que conta as varreduras **censuradas** —
logo o último quartil concentra as perguntas em que o critério nunca dispara.
Confirmação independente, pelo relógio: o `qwen3.8` passou de ~7 min para
~15 min por pergunta no último terço.

Duas consequências:

1. **Cortar por índice não é amostrar.** `--limite N` sem `--espalhar` pega o
   bloco fácil. Já tratado nos pilotos; convém ficar escrito ao lado da tabela
   de tempos da seção 5 do README.
2. **Rodada incompleta perde dado enviesado.** As perguntas que faltariam são
   sistematicamente as censuradas — justamente onde o contraste iterativo ×
   incremental acontece. Não se deve aceitar um modelo pela metade.

## 6. Contagem por direção, agora com 3 modelos

| modelo | critério | igual | incr_antes | iter_antes | só incr parou | só iter parou | ambos censurados |
|---|---|---|---|---|---|---|---|
| gpt-oss:20b | json | 171 | 10 | 4 | 6 | 0 | 1 |
| gpt-oss:20b | regex | 178 | 7 | 7 | 0 | 0 | 0 |
| mistral-small3.1 | json | 154 | 11 | 5 | 13 | 3 | 6 |
| mistral-small3.1 | regex | 163 | 12 | 4 | 8 | 1 | 4 |
| qwen3.8:27b | json | 160 | 9 | 1 | 6 | 2 | 14 |
| qwen3.8:27b | regex | 167 | 8 | 3 | 4 | 2 | 8 |

A **diluição apareceu**: `so_iterativo_parou` era 0 no `gpt-oss` e soma 8 casos
com os três modelos, contra 37 do sentido oposto. O `04_metricas.py` disparou o
aviso de que os dois sentidos ocorrem e a média de `delta_D` não resume o
contraste.

Vale lembrar que `so_*_parou` e `ambos_censurados` são **função do `D_max`**:
com `D_max = 50` o incremental tem espaço para eventualmente parar, e um caso
que com `D_max = 10` apareceria como parada exclusiva migra para as colunas
`*_antes`. Só `incremental_antes` e `iterativo_antes` comparam entre rodadas de
`D_max` diferente.

## 7. Um problema aberto no `mistral`

`json_estrito_% = 0,000` nos três modos, com `conformidade_json_% = 100,000`.
Passa no teste frouxo e reprova em **100 %** no estrito — enquanto o `gpt-oss`
dá 96–100 % e o `qwen3.8` dá 100 %. É sistemático, não ruído.

Isso precisa ser entendido antes de o `mistral` entrar em qualquer análise da
Modalidade B, porque a Modalidade B depende do formato da saída. Basta olhar
uma resposta crua dele no `b1_iteracoes_mistral_*.parquet`.

## 8. Estado dos seis modelos

| modelo | estado |
|---|---|
| `gpt-oss:20b` | ✅ concluído, flex_attention, ~7 h |
| `mistral-small3.1` | ✅ concluído, sdpa, ~9,5 h |
| `qwen3.8:27b` | ✅ concluído, sdpa, ~31 h |
| `gemma3:27b` | pico de 34,2 GB contra 31,8 da placa: pagina para a RAM |
| `MedGemma 27b` | idêntico ao `gemma3` (mesma arquitetura) |
| `gemma4:31b` | não carrega: `marcadores_fim_prefacio` inexistentes no tokenizador |

Pendências que não dependem de GPU e travam os três que faltam:

1. **Paginação** do `gemma3` e do `MedGemma`: aceitar a lentidão (~2x) ou
   quantizar o cache KV — o segundo é mudança de método.
2. **`gemma4:31b`**: pelo chat template, o canal de raciocínio é
   `<|channel>thought … <channel|>`, e com `thinking: False` o próprio prompt
   já fecha esse canal — logo `marcadores_fim_prefacio: []` é o valor correto.
   Mas o `extrair_resposta_final` não remove `<|turn>`, `<turn|>`, `<|channel>`
   nem `<channel|>` (o regex exige `<|` e `|>` juntos), e qualquer um deles
   vazaria para o BERTScore e para o regex de parada.
3. **Backends**: já são dois entre os três concluídos, e o `qwen3.8` não aceita
   `flex_attention` nesta versão do transformers. Com diferença medida entre
   backends chegando a 3,4e-1, isso precisa estar declarado no item 17.

## 9. O que muda no texto da tese

A hierarquia proposta era: η como resultado principal, S como efeito pequeno a
confirmar na replicação, `D` como nulo honesto. A replicação foi feita e o
efeito em S **não se confirmou** — e mais: a métrica que o produzia é sensível
ao comprimento, e o recall inverte o sinal em todas as células.

A frase que os dados sustentam deixa de ser *"qualidade equivalente ou melhor
a uma fração do custo"* e passa a ser *"recall de 1,3 a 5,1 pontos menor a
7–24 % do custo em tokens, no braço iterativo; o braço incremental não
apresenta ganho de custo"*. É mais defensável, e é um achado próprio: **o F1 do
BERTScore não é métrica adequada para comparar braços que produzem respostas de
comprimentos sistematicamente diferentes.**

Sugestão mínima para o `04_metricas.py`, sem custo de GPU: imprimir `P` e `R` ao
lado do `F1` e o comprimento mediano da resposta por braço. As duas colunas já
estão nas parquets.
