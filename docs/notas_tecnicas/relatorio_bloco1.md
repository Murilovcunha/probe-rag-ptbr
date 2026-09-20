# Bloco 1 — resultados e achados completos

Documento de contexto para a redação do artigo. Todos os números foram
recalculados a partir das parquets em 12/09/2026, com o mistral do braço qwen3
já corrigido. Usa a notação do manuscrito: **saturante** (bloco único, k = 50),
**iterativo** (substitui), **incremental** (acumula); eixos **D**, **S**, **η**;
**Modalidade A** (autômato regex, Eq. 2) e **Modalidade B** (contrato JSON,
Eq. 3).

---

## 0. O que existe, e o que não existe

**Desenho executado:** 3 arquiteturas × 2 recuperadores × 2 modalidades ×
3 modos de entrega × 192 consultas = **6.912 varreduras**. As 12 células
(arquitetura × recuperador × modalidade) estão completas.

| | |
|---|---|
| arquiteturas | `gpt-oss:20b`, `mistral-small3.1:24B`, `qwen3.8:27B` |
| recuperadores | `Qwen3-Embedding-8B`, `multilingual-e5-small` |
| `D_max` | 50 |
| teto de geração | 800 tokens |
| teto de *prompt* | 32.768 tokens |
| consultas | 192 (74 com *chunk*-ouro rastreável) |

**Não executado, e que precisa ir em Limitações:**

1. **`gemma3:27B`, `MedGemma 27B` e `gemma4:31B` não entraram.** O `gemma3` e o
   `MedGemma` pedem pico de 34,2 GB contra 31,8 GB disponíveis e paginam para a
   RAM; o `gemma4` não carrega por marcadores de canal ausentes no tokenizador.
   Consequência a declarar: **o único modelo com especialização médica ficou de
   fora, em corpus hospitalar.** A Tabela 2 do manuscrito tem de mostrar a
   matriz efetivamente executada, não a planejada.
2. **O baseline "contexto fixo de volume equivalente" (Seção 5.6) não foi
   executado.** É o comparador que isolaria progressividade de volume, e é o
   único imune à truncagem de geração. A nota vermelha do rascunho antecipa
   exatamente isto: precisa ir em Limitações, com a observação de que é a
   pergunta que um revisor atento fará.
3. **Não há modelo-âncora entre as duas máquinas.** O braço `e5small` rodou na
   máquina A e o braço `qwen3` na máquina B. Nenhuma célula
   arquitetura × recuperador foi executada nas duas. Detalhes em §8.

---

## 1. Validade da instrumentação — o que pode ser afirmado

Estas verificações sustentam as afirmações das Seções 5.3 e 6.1.

- **Corpus idêntico entre os dois braços.** Consulta, resposta de referência e
  arquivo-fonte batem em **100 %** das 1.152 linhas por arquitetura; apenas
  `id_chunk_top1` difere (50,5 %), que é o efeito do recuperador. A rotina de
  verificação de corpus roda no arranque e interrompe em caso de divergência.
- **`D` não depende do pós-processamento de texto.** Verificado nas 3.595
  iterações do mistral: a limpeza de tokens de controle não altera `valida`,
  `motivo_decisao`, `conformidade_json` nem `recusa_textual_no_valor` em
  **nenhuma** linha. Isso separa falha de formato de falha de ancoragem, como
  a Seção 4.4.2 exige.
- **BERTScore é reprodutível entre máquinas.** Rescorado em CPU numa terceira
  máquina, reproduz os valores da GPU original com `max|dif| = 2,4e−07`. A
  não-determinismo do projeto está na **geração**, não na medição.
- **Prompt da primeira iteração idêntico entre iterativo e incremental**
  (exigência da Seção 4.3): `W₁ = c₁` nos dois modos, verificado.

---

## 2. Seção 6.1 — Validação do pipeline de recuperação

`P@1` sobre as 74 consultas com *chunk*-ouro rastreável:

| recuperador | P@1 |
|---|---|
| Qwen3-Embedding-8B | **79,730 %** |
| multilingual-e5-small | **56,757 %** |

> **Correção necessária no manuscrito.** A Seção 5.4 registra 52,70 % para o
> `multilingual-e5-small`. O valor medido neste conjunto é **56,757 %**. O
> 79,73 % do Qwen3 confere exatamente. Vale reconciliar antes de submeter — os
> dois números do parágrafo vêm de fontes diferentes.

O contraste de recuperação é grande e é o que sustenta a Seção 6.5: o segundo
nível não é um espantalho, é um sistema alternativo publicado, de 118 milhões
de parâmetros, que recupera corretamente em mais da metade das consultas.

---

## 3. Seção 6.2 — Fidelidade sob contexto saturante

### 3.1 O braço saturante não entrega 50 páginas — e isto é resultado de primeira ordem

A Seção 5.6 já antecipa que "k = 50 posições do *ranking*" não equivale a 50
páginas. Os números:

| arquitetura | recuperador | páginas (média) | = 50 páginas | truncado pelo teto de 32.768 |
|---|---|---|---|---|
| gpt-oss | qwen3 | 37,6 | 1,0 % | **31,8 %** |
| mistral | qwen3 | 34,0 | 0,5 % | **72,4 %** |
| qwen3.8 | qwen3 | 33,1 | 0,0 % | **78,6 %** |
| gpt-oss | e5small | 38,5 | 0,5 % | **52,6 %** |
| mistral | e5small | 33,8 | 0,0 % | **88,5 %** |
| qwen3.8 | e5small | 37,8 | 0,0 % | **60,9 %** |

Duas causas independentes se somam: a deduplicação de páginas (740 segmentos
apontam para 462 páginas) reduz 50 posições a ~34–38 unidades de texto
distintas; e o teto de 32.768 tokens corta a cauda em 32 % a 88 % das consultas.

**A fração de truncagem é função do tokenizador**, e por isso o baseline não é o
mesmo objeto entre arquiteturas: o mistral recebe o contexto completo em 11,5 %
das consultas e o gpt-oss sob qwen3 em 68,2 %. Qualquer comparação
*entre arquiteturas* contra o saturante carrega essa heterogeneidade.

Recomendação: renomear no texto de "k = 50" para algo como **"contexto
saturante — as páginas distintas entre as 50 primeiras posições, até o teto de
contexto"**, e publicar esta tabela junto ao resultado.

### 3.2 Truncagem da geração, e a assimetria entre braços

Com teto de 800 *tokens*, o braço saturante trunca **~7 pontos percentuais mais**
que os braços com parada (ex.: gpt-oss/e5small/regex: 7,81 % contra 0,57 %). A
rotina de análise emite alerta automático para isso.

Direção do viés: a truncagem corta respostas longas, e S é não-monotônico no
comprimento — logo o efeito sobre o S do baseline é ambíguo, mas **existe** e
está na mesma direção em todas as células. Em versão anterior, com teto de 400,
a assimetria era 35,4 % contra 7,0 % e produzia sozinha um ΔF1 de +0,02; o teto
foi elevado para 800 por causa disso. A assimetria residual precisa constar.

### 3.3 O mecanismo: S é não-monotônico no comprimento da resposta

`S_bertscore_f1` tem máximo na faixa de razão comprimento-resposta /
comprimento-gabarito entre 0,5 e 1 (0,800), cai para 0,67 na faixa 2–4 e para
0,53 acima de 8. A decomposição mostra a causa: **a revocação se sustenta e a
precisão desaba** (0,82 → 0,45). O F1 pune verbosidade, não erro.

Razão mediana por braço (a coluna `razao_compr_mediana` da saída padrão):

| arquitetura · modalidade | saturante | incremental | iterativo |
|---|---|---|---|
| gpt-oss · A | 1,79 | 1,27 | 1,27 |
| mistral · A | 1,33 | 1,19 | 1,14 |
| qwen3.8 · A | 1,97 | 1,43 | 1,34 |
| gpt-oss · B | 1,00 | 0,83 | 0,80 |
| mistral · B | 0,92 | 0,82 | 0,73 |
| qwen3.8 · B | 1,00 | 0,88 | 0,85 |

(valores do braço qwen3; o braço e5small tem o mesmo padrão)

**O sinal do ΔF1 é previsto pela posição de cada braço nessa curva**: sob a
Modalidade A o saturante está em 1,3–2,0 (passado do ótimo) e o iterativo em
1,1–1,3 (mais perto) → ΔF1 tende a positivo; sob a Modalidade B o saturante cai
em ~1,0 (no ótimo) e o iterativo em 0,73–0,85 (curto demais) → ΔF1 inverte.
Nas três arquiteturas, nos dois recuperadores. **Não sobra efeito de método a
explicar pela via do F1.**

> Este é um **achado metodológico próprio e defensável**: o F1 do BERTScore não
> é métrica adequada para comparar braços que produzem respostas de comprimentos
> sistematicamente diferentes. É o que justifica reportar P e R separadamente,
> como a Seção 4.5.2 já prevê.

### 3.4 As 24 células de ΔS, com significância

Diferenças pareadas por consulta, n = 192, Wilcoxon.

**Iterativo − saturante:**

| arquitetura | recup. | mod. | ΔF1 | p | ΔR | p | ΔP |
|---|---|---|---|---|---|---|---|
| gpt-oss | qwen3 | A | **+0,0238** | 2,3e−05 | −0,0018 | 0,39 | +0,0436 |
| mistral | qwen3 | A | −0,0072 | 0,77 | −0,0230 | 0,028 | +0,0072 |
| qwen3.8 | qwen3 | A | **+0,0077** | 0,027 | −0,0232 | 0,0020 | +0,0309 |
| gpt-oss | e5small | A | **+0,0201** | 0,025 | −0,0134 | 0,71 | +0,0465 |
| mistral | e5small | A | −0,0068 | 0,54 | −0,0307 | 0,0083 | +0,0146 |
| qwen3.8 | e5small | A | +0,0005 | 0,39 | −0,0330 | 4,8e−05 | +0,0264 |
| gpt-oss | qwen3 | B | −0,0039 | 0,53 | −0,0311 | 0,0048 | +0,0263 |
| mistral | qwen3 | B | −0,0242 | 0,034 | −0,0358 | 0,0040 | −0,0119 |
| qwen3.8 | qwen3 | B | −0,0213 | 6,6e−04 | −0,0440 | 3,2e−09 | +0,0033 |
| gpt-oss | e5small | B | −0,0070 | 0,55 | −0,0397 | 2,6e−04 | +0,0259 |
| mistral | e5small | B | −0,0337 | 0,0047 | −0,0510 | 8,4e−05 | −0,0154 |
| qwen3.8 | e5small | B | −0,0283 | 4,5e−04 | −0,0505 | 2,5e−08 | −0,0040 |

**Leitura:**

- **ΔR é negativo em 12 de 12 células.** É o resultado mais robusto do eixo S.
- **ΔP é positivo em 10 de 12.** O ganho de F1, onde existe, é ganho de
  *precisão*, isto é, de concisão — não de correção.
- O ΔF1 positivo aparece em 4 de 12 e **sempre sob a Modalidade A**. Sob a
  Modalidade B é ≤ 0 nas seis células.

### 3.5 Análise por protocolo

Restringindo às consultas em que o braço saturante **recebeu o contexto sem
truncagem** — isto é, em que o comparador foi de fato administrado como
especificado:

| arquitetura | recup. | mod. | n | ΔF1 | p | ΔR | p |
|---|---|---|---|---|---|---|---|
| gpt-oss | qwen3 | A | 131 | **+0,0149** | 0,026 | −0,0059 | 0,95 |
| mistral | qwen3 | A | 53 | +0,0042 | 0,21 | −0,0075 | 0,92 |
| qwen3.8 | qwen3 | A | 41 | +0,0160 | 0,069 | −0,0111 | 0,80 |
| gpt-oss | e5small | A | 91 | −0,0015 | 0,90 | −0,0341 | 0,091 |
| mistral | e5small | A | 22 | −0,0396 | 0,38 | −0,0559 | 0,26 |
| qwen3.8 | e5small | A | 75 | −0,0093 | 0,90 | −0,0411 | 0,0039 |
| (Modalidade B) | | | | todas ≤ 0 | | todas < 0 | |

Este corte **não é condicionamento em variável pós-tratamento**: a truncagem é
determinada na montagem do *prompt*, antes de qualquer geração, e não depende da
saída do modelo. Excluir essas consultas é excluir unidades em que o braço de
controle não foi administrado conforme o protocolo. O custo é generalidade — o
subconjunto é outra população e o n varia de 22 a 131 —, então **as duas colunas
devem ser publicadas juntas, com a taxa de truncagem ao lado**, nunca uma no
lugar da outra.

**O que muda com o corte:** o ΔF1 de +0,0201 do gpt-oss sob e5small, que
originou a hipótese do efeito de fidelidade, vai a **−0,0015**. Sob a
Modalidade A com recuperação boa e comparador íntegro, os três modelos ficam
entre +0,004 e +0,016, mas só um atinge significância individual. **É tendência,
não efeito estabelecido.**

---

## 4. Seção 6.3 — A interrupção precoce recupera a maior parte do orçamento

η = 1 − T_inc/T_sat, calculado por consulta e agregado depois.

| arquitetura | recup. | mod. | η iterativo (média · mediana · p10 · η<0) | η incremental |
|---|---|---|---|---|
| gpt-oss | qwen3 | A | **+0,951** · 0,964 · 0,918 · 0,0 % | +0,915 · 0,964 · 0,916 · 0,5 % |
| gpt-oss | qwen3 | B | +0,899 · 0,962 · 0,847 · 2,6 % | +0,804 · 0,962 · 0,851 · 1,0 % |
| mistral | qwen3 | A | +0,832 · 0,954 · 0,828 · 4,2 % | +0,072 · 0,954 · 0,813 · 5,7 % |
| mistral | qwen3 | B | +0,737 · 0,953 · 0,713 · 8,9 % | **−0,506** · 0,953 · −0,802 · 10,9 % |
| qwen3.8 | qwen3 | A | +0,828 · 0,958 · 0,832 · 5,2 % | +0,323 · 0,958 · 0,796 · 4,7 % |
| qwen3.8 | qwen3 | B | +0,711 · 0,955 · 0,329 · 8,9 % | **−0,882** · 0,955 · 0,584 · 8,9 % |
| gpt-oss | e5small | A | +0,925 | +0,749 |
| gpt-oss | e5small | B | +0,814 | +0,336 |
| mistral | e5small | A | +0,717 | **−0,287** |
| mistral | e5small | B | +0,612 | **−0,717** |
| qwen3.8 | e5small | A | +0,760 | **−0,732** |
| qwen3.8 | e5small | B | +0,649 | **−1,457** |

**Três leituras, e a segunda é a que o manuscrito precisa:**

1. **O braço iterativo é positivo em 12 de 12 células**, com média de +0,61 a
   +0,95 e mediana sempre em ~0,95. É o resultado central do artigo e é o mais
   sólido de todos.

2. **A média e a mediana divergem sistematicamente no braço incremental**, e a
   divergência é toda cauda. A mediana fica em ~0,95 em *todas* as células,
   inclusive naquelas em que a média é −1,457. A fração com η > 0,9 fica entre
   74 % e 92 %. **Reportar só a média inverte o sinal do resultado.** Isto
   sustenta empiricamente a recusa de agregação anunciada na Seção 4.5 —
   e é um argumento melhor do que o teórico, porque é uma medida em que a
   escolha do estimador troca a conclusão.

3. **η < 0 é raro e concentrado**: 0 % a 10,9 % das consultas por célula. São as
   consultas cuja resposta exige compor informação dispersa, em que o custo
   acumulado das inferências sequenciais supera o da chamada única. É o
   compromisso anunciado na Seção 4.5.3, agora quantificado.

`censurado_limite_tokens` no incremental: 2,3 % (braço qwen3). São consultas em
que o contexto acumulado estourou o teto antes da parada — censura por memória,
não por ancoragem, e precisa ser reportada separadamente da censura à direita.

---

## 5. Seção 6.4 — Iterativo versus incremental: direção, não média

Contagens por direção, braço qwen3 (n = 192 por célula):

| arquitetura | mod. | igual | incr. antes | iter. antes | só incr. parou | só iter. parou | ambos censurados |
|---|---|---|---|---|---|---|---|
| gpt-oss | A | 188 | 2 | 2 | 0 | 0 | 0 |
| gpt-oss | B | 180 | 5 | 3 | 3 | 0 | 1 |
| mistral | A | 175 | 6 | 3 | 4 | 1 | 3 |
| mistral | B | 163 | 4 | 7 | 10 | 2 | 6 |
| qwen3.8 | A | 174 | 8 | 2 | 5 | 0 | 3 |
| qwen3.8 | B | 166 | 7 | 0 | 4 | 2 | 13 |

Braço e5small (mesmo padrão, magnitudes maiores):

| arquitetura | mod. | igual | incr. antes | iter. antes | só incr. parou | só iter. parou | ambos censurados |
|---|---|---|---|---|---|---|---|
| gpt-oss | A | 178 | 7 | 7 | 0 | 0 | 0 |
| gpt-oss | B | 171 | 10 | 4 | 6 | 0 | 1 |
| mistral | A | 163 | 12 | 4 | 8 | 1 | 4 |
| mistral | B | 154 | 11 | 5 | 13 | 3 | 6 |
| qwen3.8 | A | 167 | 8 | 3 | 4 | 2 | 8 |
| qwen3.8 | B | 160 | 9 | 1 | 6 | 2 | 14 |

**Este é o resultado que valida o desenho metodológico do artigo.** Os dois
sentidos ocorrem no mesmo experimento:

- **`so_incremental_parou`** (26 casos no braço qwen3, 37 no e5small): acumular
  **resolveu** o que uma página só não resolvia — respostas que exigem compor
  mais de um segmento;
- **`so_iterativo_parou`** (5 e 8 casos): acumular **atrapalhou** — o modelo
  deixou de reconhecer a suficiência que já tinha visto. **É diluição, medida
  diretamente.**

A rotina de análise dispara alerta automático quando os dois sentidos ocorrem,
precisamente porque a média de `delta_D` daria ≈ 0 e **esconderia os dois
efeitos**. Esse é o caso empírico que converte a Seção 4.5 de escolha estilística
em exigência metodológica: *a média é o resumo errado do contraste central*.

Uma ressalva a declarar: `so_*_parou` e `ambos_censurados` são função de
`D_max`. Com `D_max = 50` o incremental tem espaço para eventualmente parar, e
um caso que com `D_max = 10` apareceria como parada exclusiva migra para as
colunas `*_antes`. **Só `incremental_antes` e `iterativo_antes` comparam entre
rodadas de `D_max` diferente.**

---

## 6. Seção 6.5 — O achado condicional: a recuperação decide se o método importa

**Este é o resultado mais forte que o Bloco 1 produziu, e ele é novo.**

Pareando célula a célula — mesma arquitetura, mesma modalidade, só o recuperador
muda — o efeito do recuperador é **monótono em três das quatro medidas**:

| medida | melhora em | média da diferença |
|---|---|---|
| **ΔR** (iterativo − saturante) | **6 / 6** | +0,0099 |
| **η iterativo** | **6 / 6** | +0,0802 |
| **η incremental** | **6 / 6** | +0,4724 |
| ΔF1 | 5 / 6 | +0,0050 |

Seis de seis, sem exceção, em três arquiteturas e duas modalidades.

O eixo D acompanha:

| | qwen3 | e5small |
|---|---|---|
| `D` médio (iterativo, Mod. A), gpt-oss | 1,10 | 1,81 |
| `D` médio (iterativo, Mod. A), mistral | 1,75 | 2,80 |
| `D` médio (iterativo, Mod. A), qwen3.8 | 1,79 | 2,96 |
| `D = 1` (gpt-oss, Mod. A) | 94,3 % | 86,5 % |
| P@1 do recuperador | 79,73 % | 56,76 % |

**A afirmação defensável:** a profundidade de parada, o custo em *tokens* e a
perda de revocação são **co-determinados pela qualidade da recuperação**, e não
são propriedades da arquitetura geradora isoladamente. O método incremental só
apresenta vantagem de custo quando a recuperação é boa o bastante para que a
parada ocorra cedo — sob recuperação fraca, o custo acumulado de reenviar o
contexto o anula (η médio de −0,29 a −1,46).

Isto responde diretamente à pergunta anunciada na introdução: *sob que condições
a entrega progressiva compensa o custo de múltiplas inferências*. A resposta é
**condicional à força do recuperador**, e agora está medida.

Um refinamento que a análise por protocolo acrescenta: sob a Modalidade A, com
recuperação boa e comparador íntegro, a perda de revocação é de **0,6 a 1,1
ponto e indistinguível de zero** (p ≥ 0,80 nas três arquiteturas), a 83–95 % de
economia de *tokens*. Sob a Modalidade B, ou com recuperação fraca, a perda é
real e chega a 5 pontos. Cuidado na redação: "não significativo" com n de 41 a
131 é *não conseguimos distinguir*, não *é igual*.

---

## 7. Seção 6.6 — Autômato versus saída estruturada como critério de parada

### 7.1 Conformidade de formato

| arquitetura | recuperador | conformidade | JSON estrito |
|---|---|---|---|
| gpt-oss | qwen3 | 98,96–99,79 % | 95,83–99,58 % |
| mistral | qwen3 | 100 % | 97,92–99,91 % |
| qwen3.8 | qwen3 | 100 % | 100 % |
| (mesmo padrão no braço e5small) | | | |

A taxa de conformidade é métrica autônoma, como a Seção 4.4.2 exige, e o
contrato declarado em *prompt* funciona: nenhuma arquitetura fica abaixo de
95,8 % no teste estrito.

> **Nota de história do experimento que vale registrar em Limitações.** O
> `json_estrito` do mistral saiu inicialmente em **0,000 %** com conformidade em
> 100 %. Não era desobediência ao formato: era artefato de decodificação — o
> `</s>` colado após a chave de fechamento fazia o *parser* estrito falhar com
> "Extra data". A causa raiz é `skip_special_tokens=False`, necessário para o
> alinhamento de canal de outra arquitetura. **Uma taxa de conformidade de
> formato pode medir o decodificador em vez do modelo**, e a distinção só ficou
> visível porque as três leituras (estrita, objeto balanceado, regex) são
> reportadas separadamente. É um argumento a favor do desenho de três níveis
> da Seção 4.4.2.

### 7.2 A regra alternativa

`D` difere sob a regra "sentinela **ou** regex no valor" em **2 de 1.728 casos
(0,1 %)** no braço qwen3. A armadilha da Eq. 3 antecipada na Seção 4.4.3 — o
modelo que preenche a chave com uma negativa em prosa e passa como válido —
**praticamente não ocorre** nestas arquiteturas. Divergência baixa é resultado
sobre os modelos, não defeito do experimento, e fecha a questão a custo
computacional nulo.

### 7.3 A modalidade muda D e muda T

Pareado por consulta (braço qwen3):

| arquitetura | modo | Δ`D` (B − A) | `D` igual | Δ`T` |
|---|---|---|---|---|
| gpt-oss | iterativo | +0,383 | 91,5 % | +112,9 % |
| gpt-oss | incremental | +0,089 | 90,6 % | +140,7 % |
| mistral | iterativo | +0,080 | 93,7 % | +58,5 % |
| mistral | incremental | +0,754 | 85,8 % | +61,9 % |
| qwen3.8 | iterativo | +0,400 | 92,6 % | +66,2 % |
| qwen3.8 | incremental | +0,113 | 91,5 % | +174,1 % |

**A Modalidade B custa de 59 % a 174 % mais *tokens* e para mais fundo.** O
contrato de saída estruturada não é neutro em custo — é uma decisão de desenho
com preço mensurável, e o preço é grande. Isso não aparece em nenhum dos
trabalhos revisados na Seção 2, e é um achado de engenharia próprio.

E, como o Bloco 2 mostra, é a modalidade que torna a entropia circular — de modo
que a escolha de modalidade afeta simultaneamente o custo, a profundidade e a
interpretabilidade da sonda de incerteza.

---

## 8. Limitações que precisam constar

1. **Deriva entre sessões, medida.** Numa re-execução da mesma arquitetura, na
   mesma máquina, mesma imagem, com *prompts* provadamente idênticos e
   decodificação gulosa, **20,8 % das gerações do braço saturante diferiram**
   (16,2 % por contagem de *tokens*; 20,8 % por texto). O braço saturante faz
   uma única inferência e não consulta critério de parada, portanto
   pós-processamento não explica. Efeito sobre a métrica: desloca a média de F1
   de um braço em **+0,0015 ± 0,0011** — terceira casa decimal. **Os contrastes
   pareados dentro da mesma rodada são robustos** porque a deriva desloca os dois
   braços junto; o que ela obriga a declarar é a comparação **entre arquiteturas
   rodadas em sessões diferentes**, que é o caso de todas.
   Âncoras de determinismo (12 comparações, 12/12 idênticas em *hash* de
   *prompt*, de *logits* e de *tokens* gerados) confirmam determinismo **dentro
   do dia e através de reinício de processo**, inclusive com outra arquitetura
   carregada e descarregada da GPU no intervalo. A janela em que a deriva
   aparece — sessões separadas por ~12 h ou mais — permanece não testada.

2. **Máquina confundida com recuperador.** O braço `e5small` e o braço `qwen3`
   rodaram em máquinas diferentes, e não há célula executada nas duas. O efeito
   do recuperador (§6) é grande o bastante para dominar — η incremental move
   +0,47 contra uma deriva de +0,0015 em F1 —, mas o componente não controlado
   deve ser declarado. **Uma célula `gpt-oss` + `e5small` na segunda máquina
   (~7 h) resolveria isso**, e é o único item de GPU pendente do Bloco 1.

3. **Ambiente heterogêneo dentro do braço qwen3.** `gpt-oss` e `mistral` rodaram
   com `transformers` 5.17.0; `qwen3.8` com 5.16.1 (igual ao braço e5small). A
   rotina de análise detecta e alerta. Magnitude desconhecida, provavelmente
   pequena, mas não medida.

4. **Quantização heterogênea.** `gpt-oss` em precisão reduzida nativa;
   `mistral` e `qwen3.8` em 4 bits. Diferença de ordem 10⁻¹, contra 10⁻³ entre
   rotinas de atenção — já declarado na Seção 5.2, e a ordem relativa justifica
   a decisão.

5. **Rotina de atenção heterogênea.** `gpt-oss` em `flex_attention`;
   `mistral` e `qwen3.8` em `sdpa` (o `qwen3.8` não aceita `flex_attention`
   nesta versão). Diferença medida entre rotinas chega a 3,4e−01 em casos
   isolados.

6. **A régua de fidelidade é pobre por decisão de desenho.** Sem juiz por modelo
   de linguagem de classe superior, porque submeter as respostas a um serviço
   remoto violaria a restrição que motiva o trabalho. A Seção 8 já prevê
   discutir isso; os achados de §3.3 mostram que a limitação **não é neutra**:
   a métrica disponível é sensível ao comprimento, que é exatamente o eixo em
   que os braços diferem.

7. **Cortar por índice não é amostrar.** A dificuldade das 192 consultas é
   distribuída em blocos: `n_inferencias` médio por quartil de índice vai de
   2,14 a 7,27. As consultas censuradas concentram-se no último quartil —
   justamente onde o contraste iterativo × incremental acontece. **Uma rodada
   incompleta perde dado enviesado**, e `--limite N` sem espalhamento pega o
   bloco fácil.

---

## 9. Tabela 2 do manuscrito — matriz efetivamente executada

| arquitetura | repositório | quantização | atenção | pico de VRAM |
|---|---|---|---|---|
| gpt-oss:20b | `openai/gpt-oss-20b` | nativa | `flex_attention` | 16,7 GB |
| mistral-small3.1 | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 4 bits | `sdpa` | 25,7 GB |
| qwen3.8:27b | `Qwen/Qwen3.8-27B` | 4 bits | `sdpa` | 28,6 GB |

Tempos por arquitetura no Bloco 1 completo: ~7 h (gpt-oss), ~9,5 h (mistral),
~31 h (qwen3.8) sob e5small; 200 min (gpt-oss) sob qwen3 — a diferença é efeito
do recuperador sobre `D`, e é ela mesma um dado.

---

## 10. O que o Bloco 2 acrescenta (resumo, para não duplicar)

Completo nas três arquiteturas sob `e5small`, 74 consultas × 10 posições × 2
modalidades. Sustenta as Seções 6.7, 6.8 e parte da 7.2:

- **Validade:** 121.619 passos verificados, **0 divergências** de
  `argmax(logit) == token gerado`; janela alinhada em 100 %.
- **6.7:** a entropia é o pior dos cinco escores; perde para a sonda sim/não
  (p ≤ 3e−6 nas seis células) e para sobreposição lexical, que é contagem de
  palavras sem modelo. Empata com a similaridade de cosseno sob a Modalidade A
  em duas das três arquiteturas (p = 0,86 e 0,81).
- **6.8:** sob a Modalidade B a entropia é quase um substituto da decisão de
  parada — AUC de 0,955 a 0,978, o `argmin` é uma recusa em 96–100 % dos casos,
  e o posto do segmento-ouro é **significativamente pior que o acaso**
  (p ≤ 2,3e−11) nas três arquiteturas. A entropia dos *logits* passa a medir
  aderência ao formato, e não localização da informação.
- **Nula estratificada:** duas células dariam p = 0,000 na nula ingênua e vão a
  p = 0,849 e p = 0,070 quando se desconta o que a estrutura de validade explica
  sozinha. É um aviso metodológico com demonstração trabalhada.

---

## 11. Arquivos de apoio

- `consolidado_qwen3_corrigido/` — `tabela_bloco1.csv`,
  `direcao_iter_vs_incremental.csv` e `consolidado_bloco1.parquet` do braço
  qwen3, **regerados com o mistral corrigido**. Substituem os que vieram na
  entrega `qwen_ff70bb0`, que ainda continham a versão contaminada.
- O braço `e5small` já está correto na entrega `b1-b2-3modelos-FINAL_91fee4b`.
