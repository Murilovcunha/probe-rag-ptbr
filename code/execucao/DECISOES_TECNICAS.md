# Decisões técnicas da pipeline

Registro das decisões de implementação que afetam os números reportados no
artigo, com o motivo de cada uma. Todas são controladas por constantes no topo
de `comum.py`, e cada uma indica a coluna ou o metadado por onde a decisão pode
ser auditada nos arquivos de `results/`.

Este documento existe porque várias dessas escolhas não são neutras: mudam o
valor de $D$, o que entra nas médias de entropia ou o que conta como parada
válida. Quem quiser conferir um número do artigo precisa saber qual regra
estava em vigor.

---

## 1. Modelos

### 1.1 Mapeamento das tags de origem para repositórios Hugging Face

A rodada anterior usava tags do Ollama; esta usa checkpoints do Hugging Face.
A correspondência adotada:

| Tag de origem | Repositório |
|---|---|
| `gpt-oss:20b` | `openai/gpt-oss-20b` |
| `mistral-small3.1:latest` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `qwen3.8:27b` | `Qwen/Qwen3.8-27B` |

Os três geradores reportados carregam em **NF4 (4 bits)**, com uma exceção: o
`gpt-oss:20b` é carregado **sem bitsandbytes**, porque o checkpoint já é MXFP4
nativo (~13 GB) e requantizar em NF4 degradaria a qualidade sem economizar
VRAM. Como a entropia é calculada sobre *logits* e *logits* mudam com a
quantização, parte da dispersão entre modelos é atribuível a esse fator — por
isso `--quant 4bit|8bit|nativo` existe nos scripts 02 e 03, e a exceção do
gpt-oss é reportada com a magnitude ao lado.

### 1.2 Modo de raciocínio desligado em todos os modelos

Modelos que emitem raciocínio antes da resposta quebrariam os dois blocos:

- o **autômato de parada** poderia disparar sobre um raciocínio do tipo "aqui
  não há informação…" que precede uma resposta correta;
- a **entropia dos primeiros tokens** mediria a abertura do raciocínio, não a
  da resposta.

A pipeline passa `enable_thinking=False` no *chat template* e
`reasoning_effort="low"` no gpt-oss. Além disso,
`Gerador.extrair_resposta_final()` remove canais e blocos de raciocínio antes
de aplicar o autômato, e a janela de entropia do gpt-oss é alinhada ao início
do canal final (`marcadores_fim_prefacio`). Se um *template* ignorar a
*flag*, a execução emite aviso no *log*.

Se um modelo declara `marcadores_fim_prefacio` e nenhum marcador resolve para
um *id* de token, o carregamento **aborta** com mensagem explícita, em vez de
devolver posição 0 e medir silenciosamente a abertura do canal de análise — um
modo de falha que, no *log*, tem exatamente o aspecto de uma rodada bem
sucedida.

### 1.3 Determinismo

`do_sample=False` (decodificação gulosa) nos dois blocos; a semente do
embaralhamento do Bloco 2 é `42 + índice da pergunta`; os vetores das consultas
são calculados uma vez e congelados em disco, o que garante *ranking* idêntico
bit a bit entre as condições comparadas. Duas execuções do mesmo comando dão o
mesmo resultado, à parte o não-determinismo de *kernel* CUDA.

---

## 2. A modalidade de parada B (contrato JSON)

### 2.1 Prompt com medição de conformidade, não decodificação guiada

O desenho previa decodificação guiada por gramática. O `transformers` não a
traz nativamente, e acoplar um *enforcer* de gramática a arquiteturas
quantizadas em NF4 seria a principal fonte de falha da rodada. A opção adotada
foi **contrato no prompt com medição da conformidade** — que é o que a própria
definição da modalidade pede ao exigir a taxa de conformidade de formato como
métrica autônoma.

O prompt da Modalidade B repete **as mesmas regras de conteúdo** da Modalidade
A e acrescenta apenas o contrato de saída `{"resposta": "…"}`. A única
variável manipulada entre os dois critérios é o mecanismo de parada: se o texto
das instruções divergisse, a diferença observada em $D$ não seria atribuível ao
mecanismo.

A leitura da chave tem três caminhos, do estrito ao tolerante: `json.loads()`
na saída inteira; `json.loads()` no primeiro objeto balanceado (resolve cerca
de código e frases introdutórias); e expressão regular sobre a chave solta
(resolve vírgula sobrando, aspas simples e JSON truncado pelo teto de geração).
Daí saem duas colunas por iteração — `conformidade_json` (a chave foi lida?) e
`json_estrito` (a saída inteira **já era** o objeto pedido?). A segunda é a
medida de obediência ao formato; a primeira diz se o dado é aproveitável.

**Formato quebrado não trava o laço.** Quando nenhum caminho lê a chave, o
autômato da Modalidade A é aplicado ao texto bruto e a iteração fica com
`conformidade_json=False` e `motivo_decisao="fallback_regex_formato_invalido"`.
Isso mantém $D$ mensurável e deixa a separação entre *falha de formato* e
*falha de ancoragem* para a análise, que pode filtrar por essa coluna.

### 2.2 A sentinela exata, e a regra alternativa calculada de graça

A regra de insuficiência compara o valor da chave com a sentinela exata
`NÃO ENCONTRADO`. Na prática, um modelo que devolve

```json
{"resposta": "Não há informações no documento fornecido."}
```

está **recusando**, mas passa pela regra como resposta válida — e o laço para
na primeira posição, produzindo um $D$ artificialmente baixo justamente nos
modelos menos obedientes ao contrato.

A pipeline não escolhe pelo analista:

- `JSON_DECISAO = "sentinela"` (padrão) mantém a fidelidade à definição: só a
  sentinela exata conta como insuficiência, e `D_iter` segue essa regra;
- **toda** iteração grava `recusa_textual_no_valor` — o autômato da Modalidade
  A aplicado ao *valor da chave*;
- com isso, `D_iter_regra_alternativa` é calculado retroativamente sobre as
  respostas já geradas, mostrando onde o laço teria parado sob a regra
  "sentinela **ou** recusa em prosa", a custo zero de GPU;
- `04_metricas.py` reporta em quantos casos as duas regras divergem.
  Divergência alta em um modelo é resultado sobre esse modelo, não defeito do
  experimento.

Para inverter a regra basta `JSON_DECISAO = "sentinela_ou_regex"`. Para apenas
**ler** a alternativa, use a coluna `D_iter_regra_alternativa`; nada precisa ser
reexecutado.

### 2.3 A janela de entropia sob o contrato JSON

Sob o contrato, os primeiros tokens gerados são `{"resposta": "` — sintaxe, não
resposta — e é exatamente ali que a entropia seria medida. Comparar isso com a
entropia de uma frase em prosa não é comparação, é artefato.

`ALINHAR_JANELA_ENTROPIA_JSON = True` resolve reaproveitando o mecanismo que já
pulava o canal de análise do gpt-oss: `_inicio_valor_json()` decodifica os
primeiros tokens procurando o padrão `"resposta"\s*:\s*"` e devolve o índice do
primeiro token **do valor**; a janela começa ali.

Quando o modelo quebra o formato, o padrão não aparece e não há valor para onde
deslocar. Cada posição grava `janela_json_alinhada` (`True`/`False`, `None` na
Modalidade A), e daí:

- `03_bloco2_entropia.py` imprime a fração alinhada por modelo e **avisa abaixo
  de 80 %**;
- `04_metricas.py` **descarta** essas posições das análises de entropia e
  reporta quantas descartou. Elas continuam valendo para $D$, conformidade e
  formato, que não dependem da janela;
- `05_piloto.py` **falha** se a conformidade ficar abaixo de 50 % ou o
  alinhamento abaixo de 80 %, antes de comprometer uma rodada longa.

**O que continua não sendo comparável:** o *nível* de entropia das duas
modalidades, mesmo depois do alinhamento — são distribuições sobre continuações
diferentes, e o modelo sabe que está dentro de uma *string* JSON. O que se
compara entre A e B é o **posto do segmento-ouro**, invariante a reescala
monotônica, e o $D$ de cada uma.

---

## 3. O laço de entrega (Bloco 1)

### 3.1 Páginas repetidas no *ranking*

Os 740 segmentos apontam para apenas **462 páginas distintas**. Duas
constantes decorrem disso:

- `DEDUP_POR_PAGINA = False` — $D$ continua contando posições do *ranking de
  segmentos*, como no trabalho de base. Ligar a *flag* faria $D_{max}=50$
  significar 50 páginas distintas, e os números deixariam de ser comparáveis
  com aquele trabalho;
- `EVITAR_REPETIR_PAGINA_NO_CONTEXTO = True` — no modo incremental, uma página
  já presente no contexto acumulado não é concatenada de novo. Isso **não**
  altera $D$; evita apenas reenviar texto idêntico. `D_chunk` registra quantos
  segmentos distintos foram efetivamente injetados.

### 3.2 Teto de tokens de entrada (`limite_tokens_prompt = 32768`)

No modo incremental, 50 páginas somam cerca de 50 mil *tokens*. Os modelos têm
janela para isso, mas o *KV-cache* não cabe nos 32 GB da GPU junto com os pesos.
A pipeline mede o *prompt* antes de gerar e, quando estoura:

- **incremental** — encerra o modo e marca `censurado_limite_tokens=True` (o
  contexto só cresce, então estouraria em todas as iterações seguintes);
- **iterativo** — **pula** aquela posição e continua, já que cada *prompt* é
  independente, contabilizando em `n_paginas_puladas`;
- **fixo k=50** — encurta o contexto até caber e marca
  `contexto_truncado=True`.

Nada é preenchido com valor inventado. Este é o modo de falha silencioso do
braço incremental: se uma fração não trivial das consultas bate no teto, $D$
está truncado por **memória**, não por ancoragem. As duas colunas acima
permitem verificar isso, e a contagem é reportada.

### 3.3 Censura à direita

Consultas que esgotam $D_{max}=50$ sem resposta válida ficam com `D_iter` nulo
e `censurado_direita=True`. As médias de $D$ são calculadas **apenas** sobre as
não censuradas, e a contagem de censuras é reportada à parte.
**Censurado não é $D=50$**, e censurado nunca é empate.

### 3.4 Gestão de memória

Bloco 1 com `output_scores=False` e `return_dict_in_generate=False` —
obrigatório, o contexto chega a 50 páginas. Bloco 2 com registro de *logits*,
mas cada tensor reduzido a um escalar e descartado no mesmo passo do laço; o
contexto ali é de um segmento. Após **cada** inferência: `del` das saídas,
`gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, nessa
ordem — sem o `gc.collect()` no meio, o `empty_cache()` não encontra nada para
liberar, porque as saídas de `generate()` formam ciclos de referência.
`vram_pico_gb` é registrado por consulta.

---

## 4. A sonda de incerteza (Bloco 2)

### 4.1 *Logits* crus, não *scores*

`output_scores=True` devolve os valores **depois** dos *logits processors*. Com
decodificação gulosa os *warpers* são pulados, mas `repetition_penalty`,
`no_repeat_ngram_size` e `min_new_tokens` continuam entrando — e
`min_new_tokens` joga $-\infty$ na posição do EOS exatamente nos primeiros
passos, que é onde a janela de entropia cai. A distribuição medida estaria
truncada no ponto da medida, e de forma diferente em cada modelo.

A pipeline usa `output_logits=True`, e **só ele**: as duas *flags* retêm pilhas
independentes, e ligar ambas dobraria a memória sem ganho.

**O invariante de índice é contado, não asseverado.** Com decodificação gulosa,
o argmax do *logit* no passo $i$ deve ser o token gerado nesse passo. A
pipeline verifica isso em todos os passos e **conta** as divergências em vez de
abortar — uma divergência é justamente o sintoma da interferência descrita
acima, e um `assert` derrubaria uma rodada de mais de cem horas por causa do
fenômeno que a escolha de `output_logits` já neutraliza. A taxa vai para
`metadata_*.json`; taxa maior que zero é dado a reportar, não falha.

### 4.2 Janela de 20 passos como medida primária

A janela primária, declarada *a priori*, é a média dos 20 primeiros passos
úteis. O registro, porém, é completo: `entropia_por_token` guarda todos os
passos, além de *logprob*, `p_top1`, margem e o top-5 do primeiro passo. Como a
pilha de *logits* já está retida pela chamada de geração, medir todos os passos
não custa GPU adicional.

Janelas mais largas são análise **secundária pré-declarada**: servem para
mostrar se o sinal se sustenta *além* do ponto de decisão recusar/responder, que
é o que separa a hipótese de localização de um mero detector de recusa. Posições
com menos de `MIN_PASSOS_ENTROPIA = 5` passos ficam fora das médias, e a
distribuição de `n_passos_entropia` é reportada por modelo.

A normalização da entropia por $\log|V|$ foi **descartada**: os tetos diferem em
2,1 % entre os modelos, contra uma faixa observada de 0,5 a 3 nats, e a
reescala não toca no que de fato limita a comparação, que é a granularidade de
tokenização. Só o `vocab_size` fica registrado no metadado; a comparabilidade
entre modelos é resolvida por **ordenação**. Pela mesma razão, `p_top1` e a
margem, embora sejam probabilidades em $[0,1]$, não são livres de granularidade
de tokenizador: um modelo que emite `Não` como um token concentra massa que
outro distribui entre dois.

### 4.3 O posto do segmento-ouro é a estatística principal

A taxa de acerto do argmin é uma acurácia top-1 sobre $n=74$ e é **secundária**.
A comparação principal é o **posto do segmento-ouro entre as 10 posições** sob
cada escore: entropia, similaridade, *teacher forcing*, sonda sim/não e a
sobreposição lexical de controle.

É a estatística certa por dois motivos: é uma medida por consulta ($n=74$
independentes), o que permite Wilcoxon pareado direto e dispensa o *cluster
bootstrap* que um agregado de 740 pares aninhados exigiria; e é invariante a
reescala monotônica, o que torna a comparação entre modelos válida sem
normalizar nats.

**Multi-ouro:** 12 das 74 consultas têm mais de um segmento-ouro. Regra
declarada: **melhor posto entre os ouros**
(`REGRA_POSTO_MULTI_GOLD = "melhor"`). Pelo mesmo motivo, o acaso do Bloco 2 é
`n_golds/10`, não `1/10`.

### 4.4 O comparador de similaridade é o Precision@1

`max_similaridade_e_gold` não é um comparador com variância própria — é uma
identidade: o argmax de similaridade entre os 10 candidatos é sempre o de posto
1 natural, de modo que a métrica equivale a *o segmento de posto 1 é o ouro?*

Duas consequências. **(a)** A métrica é invariante ao modelo: depende só do
*ranking* do recuperador, idêntico entre os geradores por desenho. Por isso
`04_metricas.py` imprime `max_simil_e_gold_%`, `n_golds_no_top10` e
`reposicionado_%` por modelo como verificação de sanidade — as linhas têm de
bater; se não baterem, algo quebrou no *ranking*. **(b)** O corte relevante é
`gold_foi_reposicionado_no_top10`, não "o recuperador errou": sob a estatística
de posto, a similaridade só é degenerada quando o ouro foi **inserido por
substituição** — aí ele é o de menor similaridade dos dez e o posto é 10 por
construção. Com o ouro em posto 2 a 10 ele está no conjunto com a similaridade
real, e o posto continua informativo.

Daí a estratificação:

- **A_top10_natural** — comparador legítimo, comparação principal;
- **B_reposicionado** — não há comparação de similaridade possível neste
  desenho; a pergunta vira *a entropia recupera o segmento que o recuperador
  não trouxe?*, contra o acaso de 1/10, com $n$ pequeno declarado.

### 4.5 As duas sondas de um *forward* por posição

Custam de 5 % a 10 % sobre o Bloco 2 e dão três sinais independentes de
relevância por segmento em vez de um.

**Teacher forcing da resposta de referência** — *logprob* **médio por token**
da referência condicionada ao par (consulta, segmento); a soma mediria o
comprimento da referência. Dois confundidores, tratados de forma diferente:

- *afinidade estilística* — parte do *logprob* mede quanto o modelo avaliador
  gosta da fraseologia da referência. Esse termo é do par (referência, modelo),
  não varia com o segmento, entra como deslocamento aproximadamente constante
  nas 10 posições da mesma consulta e **cancela no contraste interno**. Por isso
  a estatística tem de ser o posto dentro da consulta, nunca o *logprob* bruto;
- *cópia lexical* — **este não cancela.** As referências do subconjunto
  sintético foram geradas por um modelo que estava lendo o segmento-ouro, então
  costumam citá-lo, e o *logprob* fica alto no ouro em parte por sobreposição de
  superfície — que varia com o segmento, justamente o eixo do argumento. A
  pipeline grava `sobreposicao_lexical` (token-F1 entre referência e segmento)
  como covariável, e a comparação de postos a inclui. A pergunta defensável
  passa a ser: *o posto do ouro sob o teacher forcing bate o posto do ouro sob a
  mera sobreposição lexical?*

**Sonda sim/não** — $P(\text{sim})$ para "o documento contém a resposta?". Duas
armadilhas resolvidas: no gpt-oss o passo 0 é marcador de canal, então a
pipeline força o cabeçalho do canal final dentro do *prompt* e lê o passo
seguinte; e a leitura soma a massa sobre o **primeiro token de cada variante de
superfície**, porque a negativa pode sair partida em dois tokens.
`sonda_massa_total` é gravada: massa baixa significa que o modelo não respondeu
no formato pedido e o número **não é interpretável**.

### 4.6 Nula exata e dose-resposta, sem GPU

No Bloco 2 cada *prompt* contém um único segmento, então a entropia não depende
da permutação e a nula da métrica secundária sai em forma fechada, com $V$
posições de resposta válida entre as $K=10$:

$$\mathbb{E}[\text{concordância}] = \frac{\mathbb{1}\{\text{argmin é válido}\}}{V}
\qquad
\mathbb{E}[D_{\text{embaralhado}}] = \frac{K+1}{V+1}$$

Nada de Monte Carlo. O mesmo raciocínio dá a curva dose-resposta da posição do
ouro sobre as mesmas 10 gerações, com o ouro fixo em $j$ e os outros nove
uniformes:

$$P(\text{parar no ouro} \mid \text{ouro em } j) =
\mathbb{1}\{\text{ouro válido}\}\cdot\frac{\binom{K-j}{m}}{\binom{K-1}{m}}$$

com $m$ = número de válidos entre os outros nove. Regra declarada: só consultas
com exatamente **um** ouro entre os dez entram nesta curva.

### 4.7 Rótulo alternativo por BERTScore

Com a qualidade da resposta de **cada** posição contra a referência, define-se
"segmento suficiente" e passa a existir um rótulo não anotado, com mais
positivos por consulta e sem a degeneração descrita em 4.4. Parar numa posição
diferente da anotada deixa de contar como erro quando a resposta produzida está
correta — comportamento esperado num corpus institucional, onde a mesma
informação reaparece em vários segmentos.

O limiar é um **quantil das respostas geradas sobre os segmentos-ouro
anotados** (padrão: percentil 10), não um valor absoluto: sem
`rescale_with_baseline` o BERTScore em português comprime tudo entre 0,6 e 0,9
e um corte absoluto seria arbitrário. O quantil é pré-declarável, e a
sensibilidade a ele (5 %, 10 %, 25 %) é reportada.

---

## 5. Mapeamento do segmento-ouro

Na planilha de anotação, a coluna que aponta para `corpus.id` é `id_resposta`,
no formato `"173 e 174 e 176 e 180"`. As colunas `id` e `chunk_id` daquela
planilha **não** correspondem a `corpus.id`: conferem o `nome_arquivo` em
apenas 3 das 74 linhas, por coincidência, contra **74 de 74** usando
`id_resposta`. `00_preparar_dados.py` refaz essa validação a cada execução e
**aborta** se ela deixar de valer.

`id_chunk_gold` é o primeiro *id* declarado; `ids_chunk_gold_todos` são todos.

---

## 6. Condições implementadas e não usadas no artigo

- `--prefixo-forcado` injeta um começo neutro de resposta antes de medir, para
  que a janela caia depois da bifurcação recusar/responder. Dobra o custo do
  Bloco 2 e grava em arquivos próprios, por ser condição experimental distinta
  e não variante da mesma medida;
- `DEDUP_POR_PAGINA = True` e `JSON_DECISAO = "sentinela_ou_regex"`, descritas
  acima, ficam disponíveis para quem quiser a leitura alternativa.

Nenhuma delas entra nos resultados reportados. Estão registradas aqui porque
existem no código e um leitor que abra `comum.py` vai encontrá-las.
