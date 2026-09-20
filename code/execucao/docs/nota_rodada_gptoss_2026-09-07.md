# Nota da rodada — Bloco 1, `gpt-oss:20b`, retriever `e5small`, `D_max = 50`

Registro da **primeira rodada completa** do experimento (192 perguntas) e da
avaliação que levou a subir o teto de resposta de 400 para 800 tokens.

| | |
|---|---|
| commit que rodou | `3f1cf2a` |
| retriever | `e5small` (`multilingual-e5-small`, corpus indexado SEM prefixo) |
| `D_max` | 50 |
| `MAX_NEW_TOKENS` | **400** (a rodada seguinte usa 800, commit `0314aac`) |
| escopo | 192 perguntas x 2 critérios x 3 modos |
| imagem | `logprob:1.0` `sha256:92daf569743ee0d3...` |
| ambiente | transformers 5.16.1, torch 2.14.0+cu132, vocab 201088 |

---

## 1. Tempo e interrupção

A rodada foi interrompida por um **reinício do computador** e retomada pelo
mecanismo de checkpoint.

| | |
|---|---|
| parte 1 | até a pergunta ~14 de 192, encerrada pelo reinício |
| parte 2 | 2026-09-06 19:17:46 UTC → 2026-09-07 01:37:44 UTC = **6 h 20 min** |
| total estimado | **~7 h** |
| retomada | confirmada pela mensagem `[checkpoint] retomando` |
| saída | código 0 (encerramento limpo, arquivo final gravado) |

**Causa do reinício: não determinada.** O registro de eventos do Windows não
foi consultado. Fica como pendência: se tiver sido Windows Update, metade da
rodada correu em um ambiente e metade em outro, e a versão do driver NVIDIA
antes e depois não foi verificada. Não invalida a retomada — que é
determinística e está verificada com crash simulado —, mas é um evento de
ambiente no meio de uma medição e por isso está registrado aqui.

## 2. A estimativa de tempo estava errada por ~7x

A projeção anterior era de **34–76 h por modelo** para o Bloco 1. O real foi
**~7 h**. A projeção vinha da distribuição do posto do chunk-ouro, que é
**limite superior** de `D` — e foi usada como se fosse previsão. O limite só
vale se o modelo nunca declarar suficiência antes de alcançar o gold, e ele
declara: `n_inferencias` média ficou entre 1,8 e 4,2, contra as ~7 que o
limite previa.

Consequência prática: os seis modelos do Bloco 1 cabem em cerca de uma semana,
e rodar o segundo retriever (`qwen3`) volta a ser barato — sob `qwen3` o pior
posto do gold no corpus é 10, então o loop custa ~4x menos passos.

## 3. O eixo S estava contaminado — e o teto é só metade do problema

### 3.1 Truncamento assimétrico entre os braços

Proporção de respostas com pelo menos uma inferência truncada no teto
(`n_iter_max_tokens > 0`), **por pergunta**:

| critério | fixo50 | incremental | iterativo |
|---|---|---|---|
| regex | **35,4 %** | 13,0 % | 12,5 % |
| json | 15,1 % | 3,1 % | 1,6 % |

> Atenção à definição. Estas taxas são **por pergunta**. A tabela nova do
> `04_metricas.py` calcula `n_iter_max_tokens.sum() / n_inferencias.sum()`,
> isto é, **por inferência**, e dá números menores nos braços com parada
> (regex iterativo 7,0 %, json iterativo 0,4 %) porque o denominador é maior.
> As duas leituras são válidas; o que não vale é comparar a taxa por
> inferência de um braço com a taxa por pergunta do `fixo50`, porque o
> `fixo50` faz exatamente 1 inferência e nele as duas coincidem. Na
> comparação homogênea a assimetria é 35,4 % contra 12,5 %, não contra 7,0 %.

Truncar derruba o BERTScore: no `regex fixo50`, `S_f1` = 0,724 nas respostas
inteiras contra 0,647 nas truncadas.

`S_f1` pareado por pergunta, iterativo menos fixo50:

| critério | todas (n=192) | só onde o fixo50 não truncou |
|---|---|---|
| regex | +0,0157 | **+0,0051** (n=124) |
| json | −0,0057 | **−0,0218** (n=163) |

Dois terços da vantagem aparente do iterativo sob regex vinham do teto. Sob
JSON o sinal já era negativo e fica quatro vezes mais negativo.

**Leitura defensável: o método iterativo não supera o baseline de 50 chunks em
qualidade — empata.** Para a tese isso continua sendo o resultado, porque o
empate se dá a 7 % do custo em tokens (η = 0,926 no `regex iterativo`, mediana
0,961, zero casos negativos em 192 perguntas).

Observação útil: as perguntas em que os braços com parada truncam são um
subconjunto exato das que truncam no `fixo50` (n=124 nos dois recortes). O
truncamento é propriedade da pergunta, e o pareamento restrito o neutraliza.

### 3.2 O que o teto de 800 não resolve

`S_bertscore_f1` é **não-monotônico no comprimento da resposta**. Entre as
respostas não truncadas, por razão comprimento-resposta / comprimento-gabarito:

| razão | S_f1 | P | R | n |
|---|---|---|---|---|
| < 0,5 | 0,625 | 0,726 | 0,554 | 161 |
| 0,5–1 | **0,800** | 0,824 | 0,778 | 344 |
| 1–2 | 0,771 | 0,740 | 0,808 | 309 |
| 2–4 | 0,672 | 0,602 | 0,764 | 127 |
| 4–8 | 0,591 | 0,508 | 0,708 | 42 |
| > 8 | 0,533 | 0,453 | 0,648 | 14 |

A decomposição mostra o mecanismo: quando a resposta cresce, o **recall se
sustenta** e a **precisão desaba** (0,82 → 0,45). O F1 está punindo
verbosidade, não erro.

E os braços produzem comprimentos sistematicamente diferentes. Razão mediana,
respostas não truncadas:

| braço | razão mediana | rho(comprimento, S_f1) |
|---|---|---|
| regex fixo50 | **1,61** | **−0,346** |
| regex incremental | 1,21 | −0,131 |
| regex iterativo | 1,15 | +0,042 |
| json fixo50 | 1,00 | +0,069 |
| json incremental | 0,82 | +0,056 |
| json iterativo | 0,78 | +0,223 |

O `fixo50` é o único braço já do lado descendente da curva, e o único com
correlação negativa forte. **Subir o teto dá a ele espaço para ficar ainda mais
longo, empurrando-o para as faixas 2–4 e 4–8, onde o F1 cai para 0,67 e 0,59.**
A previsão registrada aqui, antes da rodada de 800, é que o `S_f1` do
`regex fixo50` **não suba** — e possa cair.

Se cair, a conclusão "o iterativo iguala ou supera o baseline" vai parecer mais
forte depois da correção, por um motivo novo e igualmente artefatual: o
baseline penalizado por ser prolixo, não por errar.

Um número que reforça o diagnóstico: só **10,4 %** dos gabaritos passam de 1300
caracteres (o que 400 tokens produzem), mas **35,4 %** das respostas do
`fixo50` truncaram. O que estoura o teto não é a pergunta exigir resposta
longa — é o modelo ficar prolixo quando recebe 50 páginas.

**Sugestão:** que o `04_metricas.py` passe a imprimir `S_bertscore_p` e
`S_bertscore_r` ao lado do F1, e o comprimento mediano da resposta por braço.
As duas colunas já estão na parquet; não custa GPU. Sem elas não se distingue
"respondeu pior" de "respondeu mais comprido".

## 4. As colunas de direção dependem do `D_max`

Na contagem por direção do `04_metricas.py`, `so_incremental_parou`,
`so_iterativo_parou` e `ambos_censurados` são **função do `D_max`**: com
`D_max = 50` o braço incremental tem espaço para eventualmente parar, e um
caso que com `D_max = 10` apareceria como parada exclusiva migra para
`incremental_antes` / `iterativo_antes`.

Foi o que aconteceu com a pergunta 136 entre o piloto (`D_max = 10`) e a rodada
completa (`D_max = 50`): `so_iterativo_parou` = 0 nos dois critérios não
significa que a diluição desapareceu.

Só `incremental_antes` e `iterativo_antes` são comparáveis entre rodadas com
`D_max` diferente. Tabelas de piloto e de rodada completa não devem ser postas
lado a lado nas outras colunas.

## 5. Resultado que se sustenta

Na rodada de teto 400, independente das ressalvas acima:

- **η do `regex iterativo` = 0,926** de média, 0,961 de mediana, **0 % de casos
  negativos** em 192 perguntas — 93 % de economia de tokens contra o baseline
  de 50 chunks. (O truncamento do `T_fix` torna esta estimativa conservadora,
  não inflada.)
- `D` ganhou variância com o `e5small`: média 1,8–2,5 contra ~1,05 sob `qwen3`.
  **A mediana continua 1**, e iterativo e incremental dão o mesmo `D` em ~90 %
  das perguntas (171/192 json, 178/192 regex). O contraste central acontece em
  14–21 perguntas.
- **Os dois critérios discordam de sinal.** Sob JSON o incremental para antes
  com mais frequência (10 contra 4, mais 4 paradas exclusivas); sob regex é o
  iterativo (9 contra 5). É uma interação critério x modo que não estava no
  desenho e que nenhum dos pilotos podia ver.
- Modalidade B custa +119 % (incremental) e +151 % (iterativo) de tokens contra
  a A, com `delta_S` de 0,03.

## 6. Pendências

1. **Causa do reinício** — consultar o registro de eventos e comparar a versão
   do driver NVIDIA antes e depois.
2. **Rodada de teto 800** — conferir se o truncamento do `fixo50` caiu e, sobre-
   tudo, para que lado o `S_f1` do `fixo50` se moveu (ver 3.2).
3. **`P` e `R` do BERTScore no `04_metricas.py`**, e comprimento por braço.
4. **Retriever como fator** — se o Bloco 1 roda também sob `qwen3`.
5. **Regressão no `comum.py`**: o commit `0314aac` desfez o `a19ef08` e a
   docstring de `verificar_corpus_alinhado` voltou de `r"""` para `"""`, o que
   traz de volta o `SyntaxWarning: invalid escape sequence '\B'`. Inofensivo
   para os resultados, mas é sinal de que arquivos estão sendo copiados
   inteiros em vez de mesclados.

---

# Adendo — rodada de teto 800 (commit `0314aac`)

Rodada 2026-09-07, 14:38:24 → 21:34:43 UTC = **6 h 56 min**, saída 0, 192
perguntas, `D_max = 50`, `MAX_NEW_TOKENS = 800`.

## 7. O teto era um sintoma. O problema é o comprimento.

O teto de 800 fez o que se esperava com o truncamento (por pergunta):

| | 400 | 800 |
|---|---|---|
| regex fixo50 | 35,4 % | **7,8 %** |
| regex incremental | 13,0 % | 2,1 % |
| regex iterativo | 12,5 % | 1,0 % |
| json fixo50 | 15,1 % | 2,1 % |

E **não mexeu no eixo S**. `S_f1` do `regex fixo50`: 0,697 → 0,694. A previsão
registrada na seção 3.2 — de que não subiria — se confirmou, e por um caminho
mais nítido do que o previsto.

Das 68 perguntas cujo `regex fixo50` truncava em 400, 53 deixaram de truncar. A
resposta cresceu de 1300 para 1994 caracteres na mediana. O `S_f1` delas foi de
0,6469 para **0,6483**: `+0,0014`. Elas apenas trocaram a penalidade de
truncamento pela penalidade de verbosidade — caíram na faixa de razão 3,22, onde
o F1 vale 0,668.

**A diferença entre os braços é posição na curva de comprimento, não qualidade.**
Dentro de cada faixa de razão comprimento-resposta / comprimento-gabarito, os
três braços são indistinguíveis:

`S_f1` por faixa, critério regex (n entre parênteses):

| faixa | fixo50 | incremental | iterativo |
|---|---|---|---|
| < 0,5 | 0,581 (9) | 0,603 (20) | 0,609 (19) |
| 0,5–1 | 0,759 (28) | 0,746 (45) | 0,736 (53) |
| 1–2 | 0,763 (58) | 0,772 (74) | 0,777 (75) |
| 2–4 | 0,688 (51) | 0,636 (29) | 0,640 (28) |
| > 4 | 0,583 (31) | 0,585 (20) | 0,605 (15) |

Não há ordenação consistente. O que difere é a **distribuição**: 46 % das
respostas do `fixo50` têm razão > 2, contra 23 % do `iterativo`. Os braços estão
sendo pontuados em pontos diferentes de uma curva que tem máximo em razão ~0,75.

## 8. O sinal inverte no recall

| critério | métrica | fixo50 | iterativo | iterativo − fixo50 |
|---|---|---|---|---|
| regex | F1 | 0,6938 | 0,7139 | **+0,0201** |
| regex | **R** | 0,7565 | 0,7431 | **−0,0134** |
| json | F1 | 0,7510 | 0,7440 | −0,0070 |
| json | **R** | 0,7681 | 0,7284 | **−0,0397** |

O F1 diz que o iterativo ganha sob regex; o **recall diz que o baseline ganha nas
duas modalidades**. Recall é a métrica pouco sensível ao comprimento (0,78 →
0,80 → 0,76 → 0,72 ao longo das faixas, contra 0,82 → 0,73 → 0,60 → 0,53 da
precisão), e é a que responde "a resposta contém o conteúdo do gabarito?".

**Leitura que se sustenta:** o baseline de 50 chunks recupera mais conteúdo do
gabarito, nas duas modalidades. A vantagem de F1 do iterativo sob regex é efeito
de precisão causado por comprimento, não de qualidade. O resultado da tese passa
a ser um **trade-off quantificado**: o método iterativo entrega respostas com
recall 1,3 pontos (regex) a 4,0 pontos (json) menor, a **7 % do custo em tokens**
(η = 0,925). Isso é mais defensável que "qualidade igual ou melhor".

> **Correção de uma análise anterior.** O recorte "só onde o `fixo50` não
> truncou" da seção 3.1 (gap +0,0051) está **viciado**: ele condiciona no
> comprimento da resposta, que é exatamente a variável confundidora, e portanto
> seleciona o subconjunto em que o `fixo50` respondeu curto — perto do ótimo do
> F1. Com o teto de 800 o mesmo recorte dá +0,0215, não +0,005. A conclusão
> correta naquele momento seria "não identificável com estes dados".

## 9. O teto contaminava mais que o eixo S

- **`D` não é invariante ao `MAX_NEW_TOKENS`.** O critério de parada lê o texto
  gerado, e truncar cortava o sinal de parada. `D` médio do `regex incremental`:
  2,073 → 1,849; do `json incremental`: 2,296 → 2,649.
- **A contagem por direção mudou.** No regex ela foi de (5 `incremental_antes`,
  9 `iterativo_antes`) para (7, 7) — **simétrica**. A "discordância de sinal
  entre os critérios" relatada a partir da rodada de 400 era, no regex, artefato
  de truncamento. Sob JSON a assimetria persiste (10 contra 4, mais 6 paradas
  exclusivas do incremental).
- **A conformidade JSON também estava contaminada**: `json_estrito` do `fixo50`
  foi de 83,9 % para 95,8 %. Resposta truncada é JSON inválido.
- **η do `regex iterativo`**: 0,926 → 0,925, mas os casos negativos foram de 0
  para 1 em 192 (0,52 %). A afirmação "zero casos negativos" não vale mais.

## 10. O que fazer com o eixo S

O `<<< ATENÇÃO` do `04_metricas.py` disparou de novo (7 pontos percentuais).
Subir mais o teto **não** resolve — empurraria o `fixo50` ainda mais para a
direita da curva. Três caminhos, do mais barato ao mais caro:

1. **Reportar `P` e `R` ao lado do `F1`**, e o comprimento mediano por braço. As
   colunas já estão na parquet; custo zero de GPU. Sem isso não se distingue
   "respondeu pior" de "respondeu mais comprido".
2. **Reportar o `F1` estratificado por faixa de comprimento** (tabela da seção 7),
   que mostra ausência de efeito de braço.
3. **Igualar o comprimento por construção**: a mesma instrução de extensão em
   todos os braços. Resolve na origem, mas muda o desenho e obriga a refazer.

Sugestão adicional: o gatilho do `<<< ATENÇÃO` deveria testar também a
assimetria de **comprimento** entre os braços, não só a de truncamento — o
truncamento é sintoma.
