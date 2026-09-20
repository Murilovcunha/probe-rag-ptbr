# Adendo — eager vs flex_attention no Bloco 2

Complemento a `ajustes_logprob_2026-09-04.md`. Resultado do teste que você
propôs: rodar o Bloco 2 sob os dois backends e trocar o argumento pela medida.

**O que foi rodado:** `gpt-oss:20b`, 10 perguntas, 100 posições, mesma imagem,
mesmo código, mesma semente. A única diferença entre as duas execuções é a
flag `--attn`.

---

## 1. Correção: o "1e-3" que eu afirmei está errado

Na nota anterior eu escrevi que a diferença numérica entre backends era de
ordem 1e-3, e que isso era desprezível ao lado do 1e-1 do MXFP4/NF4. Afirmei
sem ter medido.

O valor real é **1,2e-2 na mediana** e **3,4e-1 no máximo** — uma ordem de
grandeza acima do que eu disse. Mesmo no subconjunto mais favorável (posições
em que os dois backends geraram texto bit-a-bit idêntico), a mediana é
**5,6e-3**.

---

## 2. Concordância entre os backends

| Métrica | Valor |
|---|---:|
| n (posições pareadas) | 100 |
| Spearman (`entropia_media`) | 0,9084 |
| Pearson | 0,8638 |
| \|erro\| mediano | 0,01243 |
| \|erro\| máximo | 0,33510 |

Correlação alta, mas não ≈ 1.

### Agregados na tela: idênticos nos dois backends

Menor entropia é o gold 40,0% · maior similaridade é gold 80,0% · posições
válidas por pergunta 4,2 · concordância de posição 30,0% · reposicionamento
0/10.

### Mas o argmin não é estável

`pos_min_entropia` mudou em **3 das 10 perguntas**:

| Pergunta | flex | eager |
|---:|---:|---:|
| 1 | 3 | 1 |
| 2 | 8 | 6 |
| 4 | 7 | 5 |

As três trocas ocorreram entre posições **não-gold**, então `min_ent_e_gold`
não mudou em nenhuma pergunta e os agregados sobreviveram. Com n=10 isso é
sorte, não estabilidade: 3 dos 10 argmins são numericamente frágeis, e a
métrica anda de 10 em 10 pontos nessa escala.

---

## 3. De onde vem o erro: decodificação, e também a atenção

| Texto gerado | Posições | Erro mediano | Erro máximo |
|---|---:|---:|---:|
| Idêntico | 45 | 0,00563 | 0,0662 |
| Divergente | 55 | 0,03321 | 0,3351 |

Duas leituras, e as duas importam.

**A decodificação gulosa mudou de caminho em 55% das posições.** Um empate
numérico flipa um token, as duas execuções seguem por textos diferentes a
partir dali, e a `entropia_media` — média sobre ~20 tokens — passa a comparar
textos distintos. É daí que vêm os erros grandes.

**Mas o piso não é zero.** Onde o texto é bit-a-bit idêntico, a entropia ainda
difere em 5,6e-3 na mediana. Isso é diferença real no cálculo da atenção.

---

## 4. O que isso faz com a métrica de argmin

Folga entre a menor e a segunda menor entropia, por pergunta (backend flex):

| | |
|---|---:|
| mediana | 0,01757 |
| média | 0,02281 |
| mínimo | 0,00313 |
| máximo | 0,06015 |

Contra o piso de ruído de 0,00563 (texto idêntico):

- razão sinal/ruído típica: **~3:1**
- o erro máximo com texto idêntico (0,0662) **excede a folga típica**

Três para um é fino para uma métrica de argmin sobre 10 candidatos quase
empatados. É evidência quantitativa a favor de promover o **posto do gold
(item 21)** sobre "menor entropia é o gold" como métrica primária — o
instrumento é seu, e ele é robusto justamente onde o argmin não é.

---

## 5. Implicação que vai além do Bloco 2

Os 55% de divergência de texto não são só um problema de entropia. No Bloco 1
o critério de parada é um autômato de regex **sobre a prosa gerada** — então
`D_iter` também depende do texto. Não afeta esta rodada, porque dentro de um
modelo o backend é fixo. Mas significa que o campo
`attn_implementation_efetiva` que você acrescentou ao metadata deixou de ser
registro burocrático: é **condição necessária para reproduzir qualquer número
do pacote**, incluindo D, T e o eixo S.

---

## 6. Decisão adotada, sujeita à sua

**Bloco 2 passa a rodar com `--attn eager`.** Três razões:

1. É a implementação de referência, e é onde a entropia é medida.
2. É mais rápido: 12,5 min contra 14,4 min para as mesmas 10 perguntas —
   prompts de ~300 tokens não compensam o custo de compilação do
   `torch.compile` que o `flex_attention` usa.
3. Elimina a pergunta do bloco que sustenta a hipótese central.

O `flex_attention` fica restrito ao **Bloco 1**, onde é obrigatório: ali o
braço fixo k=50 chega perto de 32k tokens e o eager exigiria 137 GB.

Consequência: backend por bloco em vez de por modelo, registrado no metadata
de cada um. Se preferir uniformidade, o custo é o Bloco 1 do `gpt-oss` — que
não roda de outro jeito nesta placa.

---

## 7. Limites deste teste

- 10 perguntas, 100 posições, **só no `gpt-oss:20b`**. Os outros cinco modelos
  usam SDPA e não têm essa escolha.
- As 10 perguntas são as 10 primeiras, do subconjunto sintético.
- `entropia_media` foi a única coluna comparada. `logprob_medio`,
  `margem_media` e as sondas provavelmente têm comportamento parecido, mas não
  foram medidas.

## Arquivos

`b2_entropia_gpt_oss_20b__flex.parquet`, `b2_resultados_gpt_oss_20b__flex.parquet`
(flex) e os pares sem sufixo (eager), em `comparacao_backends/`.
