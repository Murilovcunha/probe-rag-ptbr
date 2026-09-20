# Bloco 2 — sonda de entropia dos *logits*: resultados e achados completos

Documento de contexto para a redação do artigo, companheiro do
`relatorio_bloco1.md`. Números recalculados a partir das parquets em
13/09/2026, nos **dois** recuperadores. Usa a notação do manuscrito:
**Modalidade A** (autômato regex, Eq. 2) e **Modalidade B** (contrato JSON,
Eq. 3); K = 10 posições embaralhadas; a medida principal é o **posto do
segmento-ouro** entre as K posições, menor é melhor.

---

## 0. O que existe

**Desenho executado:** 3 arquiteturas × 2 recuperadores × 2 modalidades ×
74 consultas × 10 posições = **8.880 inferências com registro de *logits***.
As 12 células estão completas. O Bloco 2 usa o subconjunto de 74 consultas com
*chunk*-ouro rastreável.

| | qwen3 (`d26caf0`) | e5small (`91fee4b`) |
|---|---|---|
| linhas de entropia por arquitetura | 1.480 | 1.480 |
| consultas | 74 | 74 |
| passos verificados (total) | **164.679** | **121.619** |
| divergências de argmax | **0** | **0** |
| conformidade / JSON estrito / janela alinhada | 100 % / 100 % / 100 % | 100 % / 100 % / 100 % |
| P@1 do recuperador | 79,730 % | 56,757 % |

---

## 1. Validade da medida — o que sustenta a Seção 4.6

Estas verificações são o que autoriza tratar a entropia registrada como a
distribuição que de fato produziu o texto.

- **Invariante de consistência (§4.6.2).** `argmax(logit_i) == token gerado_i`
  em **286.298 passos somando os dois braços, com 0 divergências**. Nenhum
  processador de saída interferiu, e a decodificação gulosa é o que se diz que
  é. Uma taxa não nula seria dado a reportar; ela é exatamente zero.
- **Janela alinhada em 100 %** nas seis células de Modalidade B: a janela de 20
  passos pôde ser deslocada para depois de `{"resposta": "` em todas as
  posições, de modo que mede a resposta e não a sintaxe do contrato.
- **`inicio_janela_entropia > 0`**: 100 % no `gpt-oss` (o formato *harmony*
  sempre abre canal) e 0 % sob Modalidade A no `mistral` e no `qwen3.8` — com
  `thinking: False` não há prefácio a pular. **O aviso automático do item 03
  acusa este segundo caso como se fosse erro; é falso positivo e o texto do
  aviso precisa distinguir os dois cenários.**
- **Posições descartadas por janela curta**: 0,00–0,14 %. `n_passos_entropia`
  mediano de 8 a 11, contra os 20 declarados *a priori* — a janela primária é
  um teto, e a maioria das respostas é mais curta que ele. Vale declarar.
- **Caracterização do recuperador idêntica entre arquiteturas** dentro de cada
  braço, como o desenho exige (a Fase 1 é congelada em disco).

---

## 2. Seção 6.7 — A entropia dos *logits* localiza a evidência?

### 2.1 A tabela principal (item 21)

Posto do segmento-ouro entre 10 posições, média sobre 74 consultas.

**Recuperador Qwen3-Embedding-8B:**

| arquitetura | mod. | **entropia** | similaridade | sonda sim/não | logprob ref.* | sobrep. lexical* |
|---|---|---|---|---|---|---|
| gpt-oss | A | **4,500** | 1,527 | 1,446 | 1,203 | 1,203 |
| mistral | A | **3,419** | 1,527 | 1,324 | 1,162 | 1,203 |
| qwen3.8 | A | **2,378** | 1,527 | 1,324 | 1,149 | 1,203 |
| gpt-oss | B | **7,649** | 1,527 | 1,446 | 1,203 | 1,203 |
| mistral | B | **8,446** | 1,527 | 1,324 | 1,162 | 1,203 |
| qwen3.8 | B | **9,095** | 1,527 | 1,324 | 1,149 | 1,203 |

**Recuperador multilingual-e5-small:**

| arquitetura | mod. | **entropia** | similaridade | sonda sim/não | logprob ref.* | sobrep. lexical* |
|---|---|---|---|---|---|---|
| gpt-oss | A | **5,108** | 3,473 | 1,392 | 1,878 † | 1,311 |
| mistral | A | **3,622** | 3,473 | 1,392 | 1,189 | 1,311 |
| qwen3.8 | A | **3,595** | 3,473 | 1,446 | 1,189 | 1,311 |
| gpt-oss | B | **8,608** | 3,473 | 1,392 | 1,878 † | 1,311 |
| mistral | B | **9,203** | 3,473 | 1,392 | 1,189 | 1,311 |
| qwen3.8 | B | **9,486** | 3,473 | 1,446 | 1,189 | 1,311 |

\* `logprob ref.` e `sobreposição lexical` usam o gabarito: são **oráculos**, e
entram como teto de desempenho, não como alternativas implantáveis. A
comparação que decide é com a **sonda sim/não**, que opera no mesmo cenário da
entropia — sem resposta de referência, um *forward* por segmento.

† Valor subestimado por defeito de instrumentação corrigido depois desta
rodada; ver §7.

### 2.2 Wilcoxon pareado, n = 74 por célula

`delta > 0` = a entropia põe o segmento-ouro mais fundo, isto é, é pior.

**Contra a sonda sim/não — perde nas 12 células:**

| mod. | arquitetura | qwen3 | e5small |
|---|---|---|---|
| A | gpt-oss | +3,05 · p=1,8e−08 | +3,72 · p=4,7e−09 |
| A | mistral | +2,09 · p=3,5e−07 | +2,23 · p=1,9e−06 |
| A | qwen3.8 | +1,05 · p=5,6e−04 | +2,15 · p=2,9e−06 |
| B | gpt-oss | +6,20 · p=2,9e−13 | +7,22 · p=9,1e−14 |
| B | mistral | +7,12 · p=1,4e−13 | +7,81 · p=4,8e−14 |
| B | qwen3.8 | +7,77 · p=1,8e−14 | +8,04 · p=1,1e−14 |

**Contra a similaridade de cosseno — e aqui o recuperador muda o resultado:**

| mod. | arquitetura | qwen3 | e5small |
|---|---|---|---|
| A | gpt-oss | +2,97 · p=2,5e−08 | +1,64 · p=3,6e−03 |
| A | mistral | +1,89 · p=2,4e−06 | **+0,15 · p=0,86** |
| A | qwen3.8 | +0,85 · p=1,0e−02 | **+0,12 · p=0,81** |
| B | (as três) | +6,12 a +7,57 · p ≤ 4,2e−13 | +5,14 a +6,01 · p ≤ 2,1e−10 |

Sob `e5small` a entropia **empatava** com a similaridade em duas células da
Modalidade A. Sob `qwen3` esses empates desaparecem: perde nas seis, com
p ≤ 1,0e−02. O motivo é que o comparador melhorou — o posto da similaridade vai
de 3,473 para 1,527. **Quanto melhor o recuperador, mais claramente a entropia
perde para o escore do próprio recuperador.**

Perde também para a **sobreposição lexical** — contagem de palavras em comum
entre segmento e gabarito, sem modelo nenhum — nas 12 células.

### 2.3 A entropia bate o acaso?

Com K = 10, um ordenador aleatório dá posto esperado 5,5. Wilcoxon de uma
amostra:

| mod. | arquitetura | qwen3 | e5small |
|---|---|---|---|
| A | gpt-oss | 4,500 · melhor · p=1,0e−02 | 5,108 · **não difere** · p=0,25 |
| A | mistral | 3,419 · melhor · p=9,8e−07 | 3,622 · melhor · p=1,4e−05 |
| A | qwen3.8 | 2,378 · melhor · p=3,7e−11 | 3,595 · melhor · p=8,2e−06 |
| B | gpt-oss | 7,649 · **pior** · p=1,7e−08 | 8,608 · **pior** · p=2,1e−11 |
| B | mistral | 8,446 · **pior** · p=1,5e−11 | 9,203 · **pior** · p=1,0e−13 |
| B | qwen3.8 | 9,095 · **pior** · p=7,1e−14 | 9,486 · **pior** · p=5,8e−15 |

Sob a Modalidade A, a entropia supera o acaso em 5 das 6 células — e sob o
recuperador competente, nas 3. Sob a Modalidade B é **significativamente pior
que o acaso** nas 6: não é ausência de sinal, é **sinal invertido**.

Pela estatística secundária de `argmin` (a fração em que a menor entropia
aponta o segmento-ouro), contra o acaso de ~11 %:

| mod. | qwen3 (gpt-oss / mistral / qwen3.8) | e5small |
|---|---|---|
| A | 33,8 % / 43,2 % / **66,2 %** | 35,1 % / 48,6 % / 52,7 % |
| B | 1,4 % / 1,4 % / 0,0 % | 2,7 % / 1,4 % / 0,0 % |

### 2.4 A conclusão que os dados sustentam

> Sob a Modalidade A e recuperação competente, a entropia dos *logits* carrega
> sinal real sobre a localização da evidência: supera o acaso nas três
> arquiteturas (p ≤ 1,0e−02) e o `argmin` acerta o segmento-ouro de 3 a 6 vezes
> acima do acaso. Mas ela é **dominada por todos os comparadores** — inclusive
> por uma linha de base lexical sem modelo nenhum e pela similaridade do próprio
> recuperador —, e sob a Modalidade B o sinal se inverte.

Esta formulação é mais defensável do que "a entropia não se sustenta como
sonda de ancoragem". *Tem sinal, mas é dominada até por contagem de palavras* é
uma afirmação mais difícil de contestar do que um nulo puro, e não depende de o
revisor aceitar um resultado negativo.

---

## 3. Seção 6.8 — Sob saída estruturada, a entropia torna-se circular

### 3.1 O diagnóstico bilateral

AUC de "a recusa tem entropia menor que a resposta válida":

| arquitetura | mod. B — qwen3 | mod. B — e5small | mod. A — qwen3 | mod. A — e5small |
|---|---|---|---|---|
| gpt-oss | **0,987** recusa menor | **0,978** recusa menor | 0,598 recusa menor | 0,577 recusa menor |
| mistral | **0,927** recusa menor | **0,955** recusa menor | 0,380 **válida** menor | 0,314 **válida** menor |
| qwen3.8 | **0,964** recusa menor | **0,975** recusa menor | 0,278 **válida** menor | 0,359 **válida** menor |

E a fração em que o `argmin` da entropia é uma **recusa**:

| | qwen3 | e5small |
|---|---|---|
| Modalidade B | 94,6 % / 95,9 % / **100 %** | 97,3 % / 95,9 % / **100 %** |
| Modalidade A | 45,9 % / 41,9 % / 24,3 % | 58,1 % / 39,2 % / 41,9 % |

**Sob a Modalidade B, o mínimo de entropia é uma recusa praticamente sempre, em
três arquiteturas e dois recuperadores.** Deixa de ser característica de um
modelo e passa a ser **propriedade da modalidade de saída**.

O mecanismo é direto: o contrato pede um objeto JSON e a sentinela é uma string
fixa. Emitir a sentinela é a continuação mais previsível que existe — entropia
próxima de zero por construção. Emitir uma resposta substantiva é
comparativamente incerto. A entropia passa a medir **aderência ao formato**, e
não localização da informação. Daí o posto pior que o acaso da §2.3: o `argmin`
seleciona sistematicamente uma posição que não sustenta a resposta.

Sob a Modalidade A não há circularidade, mas **também não há direção
consistente**: o sinal inverte entre o `gpt-oss` e os outros dois, nos dois
recuperadores. Isso é resultado sobre os modelos, e precisa ser reportado como
tal.

### 3.2 O alerta bilateral não é decoração

O diagnóstico monitora os dois lados, como a Seção 4.6.4 anuncia. O lado AUC→1
é o que apareceu (a recusa tem entropia menor, e o escore vira detector de
recusa). O lado **AUC→0 é mais perigoso**: quando o segmento-ouro é a única
posição capaz de sustentar uma resposta, ele é apontado *por construção* e a
hipótese central sai confirmada sem que o escore tenha dito nada. As células de
Modalidade A com `valida_menor` (AUC 0,278 a 0,380) estão nessa direção, e é
por isso que elas **não** devem ser lidas como "a entropia funciona melhor
aqui". `separacao_total` é `False` em todas as 12 células, então nenhuma é
degenerada — mas a proximidade importa e deve ser declarada.

### 3.3 O contraste que separa bem, e é monotônico

O item 08 — massa de probabilidade de recusa no primeiro passo — separa
`não-gold` de `gold` sob a Modalidade B com margem larga (0,867–0,956 contra
0,100–0,205 no braço e5small) e é **monotônico**, ao contrário da entropia, que
tem forma em U. Sob a Modalidade A quase desaparece. É o candidato natural a
sonda, se o trabalho futuro quiser uma.

---

## 4. A nula estratificada — a contribuição metodológica

### 4.1 Por que a nula ingênua é inadequada

Validade e incerteza são **duas consequências da relevância**, não uma
mediadora da outra. Uma nula que ignora o acoplamento entre elas pode tanto
esconder sinal quanto atribuir a um escore uma anti-informatividade que é, na
verdade, estrutura de validade. A nula estratificada permuta os escores
**dentro de cada estrato de posições que produziram resposta válida**,
preservando exatamente o acoplamento observado e testando só o resíduo.

### 4.2 O que ela desfez, e o que ela deixou de pé

| mod. | arquitetura | p ingênua (posto) | p estratificada (posto) — e5small | p estratificada — qwen3 |
|---|---|---|---|---|
| A | gpt-oss | 0,261 | 0,019 | **1,0e−04** |
| A | mistral | **0,000** | **0,849** | **0,0077** |
| A | qwen3.8 | **0,000** | 0,070 | **5,0e−05** |
| B | gpt-oss | 1,000 | 0,104 | 0,017 |
| B | mistral | 1,000 | 1,000 | 0,988 |
| B | qwen3.8 | 1,000 | 1,000 | 1,000 |

**Sob `e5small`, o `mistral` e o `qwen3.8` na Modalidade A dariam p = 0,000 na
nula ingênua — dois resultados "altamente significativos" que evaporam
(p = 0,849 e p = 0,070) quando se desconta o que a estrutura de validade
explica sozinha.** É a demonstração trabalhada do argumento, e é forte porque o
caso do `mistral` é uma inversão completa: de p = 0,000 para p = 0,849.

Sob `qwen3` a estatística sobrevive nas três células de Modalidade A. Isso não
enfraquece o argumento metodológico — a nula ingênua continua produzindo
significância onde a estratificada não produz —, e reforça a conclusão de §2.4:
há sinal, e ele é real, mas é dominado.

Pela estatística de `argmin` a leitura é menos dura, e as duas **devem ser
reportadas juntas porque discordam**: sob `qwen3`, p_argmin_estratif de 5,0e−05,
0,0010 e 5,0e−05 nas três células de Modalidade A.

---

## 5. Concordância entre o eixo D e o eixo da entropia

| mod. | arquitetura | observada — qwen3 | nula | observada — e5small | nula |
|---|---|---|---|---|---|
| A | gpt-oss | 20,3 % | 12,1 % | 24,3 % | 23,1 % |
| A | mistral | 36,5 % | 27,7 % | 45,9 % | 42,1 % |
| A | qwen3.8 | 39,2 % | **45,3 %** | 47,3 % | 43,7 % |
| B | gpt-oss | 0,0 % | 0,3 % | 1,4 % | 1,9 % |
| B | mistral | 2,7 % | 0,4 % | 4,1 % | 0,7 % |
| B | qwen3.8 | 0,0 % | 0,0 % | 0,0 % | 0,0 % |

Diferenças de 0 a 8 pontos percentuais sobre a nula exata, e **uma célula abaixo
da nula**. Os dois eixos não se informam mutuamente. Isso sustenta duas leituras
do manuscrito: `D` como nulo honesto, e a entropia como medida de conformidade
mais do que de ancoragem. Vale notar que a nula aqui é **exata**, obtida em
forma fechada graças à independência da medida em relação à permutação
(Seção 4.6.1) — não é simulação.

---

## 6. O fator experimental atravessa os dois blocos

A Seção 6.5 do manuscrito estabelece, no Bloco 1, que a força da recuperação
decide se o método importa. **O Bloco 2 mostra o mesmo fator operando sobre a
sonda de incerteza**, o que eleva o recuperador de detalhe experimental a eixo
de análise:

| efeito do recuperador (qwen3 vs e5small) | |
|---|---|
| posto da entropia, Modalidade A | melhora nas 3 (5,11→4,50; 3,62→3,42; 3,60→2,38) |
| posto da similaridade | 3,473 → 1,527 |
| empates entropia × similaridade | 2 de 6 → **0 de 6** |
| nula estratificada, Modalidade A | sobrevive em 1 de 3 → **3 de 3** |
| entropia bate o acaso, Modalidade A | 2 de 3 → **3 de 3** |
| circularidade, Modalidade B | replica: AUC 0,955–0,978 → 0,927–0,987 |

**A circularidade é o único achado invariante ao recuperador** — o que é
coerente com o mecanismo: ela decorre do contrato de saída, e não da
recuperação. Todo o resto se move.

### 6.1 Um efeito colateral do recuperador competente

Sob `qwen3` **não existe o estrato `B_reposicionado`**: nas 74 consultas o
segmento-ouro já está entre as 10 primeiras posições, e não foi preciso inseri-lo
por substituição. Sob `e5small`, 13 de 74 (17,6 %) precisaram.

Isso tem duas consequências:

1. A análise por estrato (item 22c) **só é possível no braço `e5small`**.
2. O estrato `B_reposicionado` é precisamente onde a similaridade não é
   comparador legítimo (posto = 10 por construção) e onde a pergunta passa a ser
   *o escore de incerteza recupera o segmento que o recuperador não trouxe?*.
   A resposta: a entropia dá 3,385 a 9,615 e a sonda dá 1,308 a 1,846.
   **Mesmo no estrato desenhado para dar à entropia a sua melhor chance, ela
   perde para a sonda.** Vale reportar — é o teste mais adversarial disponível.

---

## 7. Dois defeitos de instrumentação encontrados e corrigidos

Registrados aqui porque afetam a leitura das duas tabelas da §2.1 e porque a
correção é ela própria um resultado sobre o desenho.

**(a) `logprob_referencia` não forçava o cabeçalho de canal.** A sonda sim/não
já o fazia; o cálculo do *logprob* da referência, não. O `gpt-oss` é a única
arquitetura com `prefixo_canal_final` não nulo, logo a única afetada: o gabarito
era pontuado numa posição em que o *template* espera abrir um canal.

| mediana de `logprob_medio_referencia` | e5small (com defeito) | qwen3 (corrigido) |
|---|---|---|
| gpt-oss | **−8,572** | **−2,708** |
| mistral | −2,013 | −1,975 |
| qwen3.8 | −2,237 | −2,202 |

O posto do oráculo do `gpt-oss` foi de 1,878 para 1,203, entrando na faixa
estreita dos outros dois (1,149–1,203). Que o *patch* seja quase nulo nas
arquiteturas sem canal é a confirmação de que ele age onde deve.
**Consequência para o texto:** a célula 1,878 da tabela `e5small` é limite
inferior e precisa de nota; a tabela `qwen3` está correta.

**(b) O teste pareado do item 21 não rodava.** Um desempacotamento de três
chaves em duas, com `except` largo demais, fazia o erro parecer
incompatibilidade de biblioteca e derrubava junto a comparação A × B. As
estatísticas das §2.2, §2.3 e §8 vêm da reanálise, sem custo de GPU.

Nenhum dos dois afeta a entropia, que é a medida central — mas os dois afetam
comparadores, e o primeiro afeta um número que ia para a tabela.

---

## 8. Modalidade A × B — o custo do contrato de saída

Posto do segmento-ouro sob a entropia, pareado por consulta (é a única
estatística de entropia comparável entre modalidades, por ser invariante a
reescala monotônica):

| arquitetura | qwen3: A → B | delta | p | e5small: A → B | delta | p |
|---|---|---|---|---|---|---|
| gpt-oss | 4,50 → 7,65 | +3,15 | 1,1e−07 | 5,11 → 8,61 | +3,50 | 1,9e−08 |
| mistral | 3,42 → 8,45 | +5,03 | 8,9e−12 | 3,62 → 9,20 | +5,58 | 9,3e−12 |
| qwen3.8 | 2,38 → 9,09 | +6,72 | 1,7e−13 | 3,59 → 9,49 | +5,89 | 3,5e−12 |

**A modalidade de saída piora o posto em 3,2 a 6,7 posições de 10, nas seis
combinações, com p ≤ 1,9e−07.** É o maior efeito medido no Bloco 2 — maior que
qualquer diferença entre arquiteturas e maior que o efeito do recuperador.

Somado ao que o Bloco 1 mostra — a Modalidade B custa de 59 % a 174 % mais
*tokens* e para mais fundo —, o contrato de saída estruturada aparece como uma
decisão de desenho com **três preços simultâneos**: custo, profundidade e
interpretabilidade da sonda de incerteza. Nenhum dos trabalhos revisados na
Seção 2 documenta isso.

---

## 9. Limitações específicas do Bloco 2

1. **`MedGemma 27B` fora.** O `MODELOS_BLOCO2` original previa `gpt-oss`,
   `mistral`, `gemma3:27B` e `MedGemma 27B`. Rodaram os três que tinham Bloco 1.
   Em corpus hospitalar, a ausência do único modelo com especialização médica é
   uma limitação de escopo que precisa ser declarada, não descoberta pelo
   revisor.
2. **A janela de 20 passos é um teto, não uma média.** `n_passos_entropia`
   mediano de 8 a 11. A janela foi fixada *a priori*, o que é a decisão certa,
   mas o número efetivo deve ser reportado.
3. **A caracterização por estrato só existe no braço `e5small`** (§6.1).
4. **`so_*` e a estatística de posto dependem de K = 10.** Com K maior, o acaso
   muda e os postos não são comparáveis entre configurações.
5. **Máquina confundida com recuperador**, como no Bloco 1: o braço `e5small`
   rodou na máquina A e o `qwen3` na máquina B, sem célula-âncora comum. Aqui o
   risco é menor que no Bloco 1, porque os efeitos medidos (deltas de 1 a 7
   posições, p ≤ 1e−06) são de outra ordem que a deriva observada — mas o
   componente não controlado deve constar.
6. **Determinismo.** Âncoras de *hash* de *prompt*, de *logits* e de *tokens*
   gerados: 12/12 idênticas no braço e5small e 12/12 no par do `qwen3.8` do
   braço qwen3. Confirma determinismo dentro do dia e através de reinício de
   processo. A janela entre sessões separadas por mais de 12 h, em que a deriva
   foi observada no Bloco 1, permanece não testada.

---

## 10. O que o Bloco 2 entrega ao artigo, em três frases

1. **A entropia dos *logits* tem sinal, e é dominada.** Supera o acaso sob a
   Modalidade A com recuperação competente, mas perde para a sonda sim/não, para
   a similaridade do recuperador e para uma contagem de palavras sem modelo, nas
   12 células.
2. **Sob contrato de saída estruturada ela se torna quase um substituto da
   decisão de parada** — AUC de 0,927 a 0,987, `argmin` é recusa em 94,6–100 %,
   posto significativamente pior que o acaso — e passa a medir aderência ao
   formato, não localização da informação. É o único achado do Bloco 2 invariante
   ao recuperador.
3. **A nula ingênua fabrica significância** onde a estratificada por validade
   não encontra: duas células vão de p = 0,000 para p = 0,849 e p = 0,070. É um
   aviso metodológico com demonstração trabalhada, transferível a qualquer
   trabalho que use sinais de incerteza como evidência sobre relevância.

Os três replicam em três arquiteturas. O (2) é o mais forte e é o que eu
colocaria como contribuição principal do bloco.
