# Adendo 2 — reprodutibilidade da medição do Bloco 2

Substitui a versão anterior deste arquivo, que afirmava que a medição "não
reproduz entre sessões". A afirmação era forte demais: **ela reproduz bit a
bit**, inclusive através de um reinício completo da máquina. O que existe é
uma deriva pontual, ainda não explicada, entre duas janelas separadas por
cerca de doze horas.

Todos os testes com `gpt-oss:20b`, Bloco 2, 5 perguntas, 50 posições.

---

## 1. O que é reprodutível

**Execuções consecutivas na mesma sessão.** Duas rodadas `eager` seguidas
deram resultado idêntico; duas `flex_attention` seguidas, também.

**Através de reinício do Windows.** Rodada antes de reiniciar, rodada depois:

| | |
|---|---:|
| posições comparadas | 50 |
| mesmo `id_chunk` | 50 / 50 |
| posições com \|erro\| > 1e-6 | **0 / 50** |
| \|erro\| mediano e máximo | **0,0** |
| `pos_min_entropia` por pergunta | idêntico nas 5 |

Erro exatamente zero, atravessando recarga do driver, reinício do WSL e do
Docker. O pipeline é determinístico.

---

## 2. Correção ao adendo 1 — e correção da minha correção

A versão anterior deste documento dizia que a diferença medida entre `eager` e
`flex_attention` era, na verdade, variação entre execuções. **Isso está
errado.** As duas rodadas do adendo 1 foram consecutivas, na mesma sessão, no
mesmo bloco de comandos. A comparação era válida.

O número do adendo 1 permanece: diferença entre backends com mediana de
**0,0124** e máximo de 0,335.

O que se acrescenta é um segundo fenômeno, de magnitude parecida e origem
diferente — a deriva descrita a seguir. Foi a semelhança de magnitude que me
levou a confundir os dois.

---

## 3. A deriva não explicada

Duas execuções do **mesmo backend, mesmo código, mesma imagem**, separadas por
cerca de doze horas (04/09 à noite × 05/09 de manhã):

| | |
|---|---:|
| mesmo `id_chunk` | 50 / 50 |
| mesmo `n_tokens_prompt` | 50 / 50 |
| **mesmo `texto_resposta`** | **23 / 50** |
| \|erro\| mediano na entropia | 0,0135 |
| \|erro\| máximo | 0,323 |

Recuperação idêntica e prompt idêntico token a token — e ainda assim 27 das 50
posições geraram texto diferente. As posições de mínima entropia foram de
`3, 8, 3, 7, 2` para `1, 10, 3, 4, 2`.

### Suspeitos eliminados por teste

| Suspeito | Como foi descartado |
|---|---|
| Código | `git checkout` da versão anterior reproduz o resultado NOVO |
| Imagem Docker | `sha256:92daf569…` idêntico nas duas janelas |
| Driver NVIDIA | 610.43.02 nas duas |
| Kernel do WSL / Docker | 6.18.33.2-microsoft-standard-WSL2 / 29.6.2 |
| Embeddings e corpus | mesmos arquivos; `id_chunk` e `n_tokens_prompt` batem |
| Recuperação | 50/50 idêntica |
| Não-determinismo do backend | execuções consecutivas batem bit a bit |
| Reinício da máquina | erro exatamente 0,0 (seção 1) |
| Cache do `HF_HOME` | único arquivo alterado na janela é telemetria do `huggingface_hub` |

**Não identifiquei a causa.** Como o cálculo é determinístico e os insumos
verificados são idênticos, alguma coisa no ambiente mudou uma vez naquela
janela. A busca chegou ao retorno decrescente e foi interrompida.

---

## 4. Por que a conclusão prática não depende da causa

A deriva não é do tempo passando nem do reinício: é de uma mudança de
ambiente. Isso é gerenciável, e a mitigação é operacional:

**Rodar os seis modelos numa janela com o ambiente congelado.** Windows Update
pausado, atualização automática do driver NVIDIA desligada, Docker Desktop sem
auto-update, e nenhuma sondagem nova de kernels no meio da rodada.

Dentro dessa janela, os seis modelos são medidos no mesmo instrumento, que é o
que a comparação entre eles exige. Cada modelo já roda de ponta a ponta numa
sessão, então internamente a consistência está garantida de qualquer forma.

---

## 5. O que continua valendo do adendo 1

A folga entre a menor e a segunda menor entropia, por pergunta, tem mediana de
**0,0176**. A diferença entre backends tem mediana de **0,0124**, e a deriva
entre janelas, **0,0135**.

As três grandezas são da mesma ordem. A métrica de argmin — "menor entropia é
o gold" e `pos_min_entropia` — depende de separar candidatos por uma margem
comparável a qualquer perturbação do ambiente. Isso é independente da causa da
deriva, e sustenta a sugestão de promover o **posto do gold (item 21)** a
métrica primária.

---

## 6. Alcance no Bloco 1

O critério de parada da Modalidade A é um autômato de regex sobre a prosa
gerada. Se a geração muda, `D_iter` e `T` mudam junto. Dentro de uma rodada
isso é consistente; entre modelos rodados em janelas de ambiente diferentes,
não necessariamente. A mitigação da seção 4 cobre os dois blocos.

---

## 7. Limites

- Só o `gpt-oss:20b`, em MXFP4 nativo. Modelos NF4 não foram testados.
- 5 perguntas, 50 posições, do subconjunto sintético.
- A estabilidade do `eager` entre janelas longas não foi medida — as duas
  execuções `eager` foram consecutivas.

Os parquets de todas as execuções citadas estão em `comparacao_backends/`.
