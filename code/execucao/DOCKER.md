# Execução em contêiner (Docker)

Alternativa ao ambiente virtual da seção 2 do `README.md`. Fixa a versão do
PyTorch, do CUDA e de todas as dependências numa imagem, o que torna a rodada
reproduzível em outra máquina sem repetir a negociação de versões que a seção
2.2 descreve.

O restante do `README.md` continua valendo integralmente: os comandos dos
scripts, o piloto obrigatório da seção 4.6, o comportamento de checkpoint da
seção 4.7 e o que devolver da seção 7 são idênticos — só o prefixo do comando
muda.

---

## 1. Pré-requisitos do host

| Item | Versão | Observação |
|---|---|---|
| Driver NVIDIA | ≥ 580 | exigido pelo runtime CUDA 13.x da imagem |
| Docker Engine | ≥ 24 | com o plugin `docker compose` v2 |
| NVIDIA Container Toolkit | ≥ 1.17 | é o que expõe a GPU ao contêiner |
| Disco livre | ≈ 320 GB | no volume apontado por `DIR_HF_HOST` |

O CUDA **não** precisa estar instalado no host: a imagem traz o runtime, e o
driver é o único componente compartilhado. Confirme que o toolkit está ativo:

```bash
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
```

Se este comando não listar a GPU, pare aqui — nenhum experimento vai rodar.

---

## 2. Configuração

```bash
cp .env.exemplo .env
```

Edite `.env` e preencha:

- `DIR_HF_HOST` — pasta onde os ~300 GB de pesos do Hugging Face serão
  gravados. Fica fora da imagem de propósito: assim `docker compose build`
  pode ser refeito sem perder o download.
- `DIR_SAIDAS_HOST` — pasta onde a `saidas/` do experimento será gravada.
- `UID` / `GID` — resultado de `id -u` e `id -g`. Sem isso os arquivos
  gerados saem pertencendo ao root.
- `HF_TOKEN` — opcional; veja a seção 4.

```bash
docker compose build
```

---

## 3. Onde cada coisa fica

| No host | No contêiner | Conteúdo |
|---|---|---|
| a própria pasta do projeto | `/app` | código e `dados/` de entrada |
| `DIR_HF_HOST` | `/hf` | cache de pesos (`HF_HOME`) |
| `DIR_SAIDAS_HOST` | `/saidas` | resultados (`LOGPROB_SAIDAS`) |

A pasta do projeto é montada, não copiada: uma alteração num `.py` no host
vale na próxima execução, sem rebuild. O rebuild só é necessário quando o
`requirements.txt` mudar.

Os scripts leem `LOGPROB_DADOS` e `LOGPROB_SAIDAS` do ambiente, já definidos
no `docker-compose.yml` — não é preciso passar caminho nenhum na linha de
comando.

---

## 4. Autenticação no Hugging Face

Quatro dos seis modelos são *gated* (seção 2.4 do `README.md`): aceite a
licença nas páginas dos modelos com a sua conta antes de tentar baixar.

Depois, escolha uma das duas formas:

```bash
# a) token no .env  -> HF_TOKEN=hf_xxxxxxxx

# b) login interativo, uma vez; o token fica em /hf e persiste
docker compose run --rm logprob hf auth login
```

---

## 5. Execução

Cada comando do `README.md` vira `docker compose run --rm logprob <comando>`:

```bash
# verificação de ambiente (seção 2.5)
docker compose run --rm logprob python verificar_ambiente.py

# teste de fumaça, sem GPU e sem download (seção 4.1)
docker compose run --rm logprob python 01_embeddings_perguntas.py --smoke
docker compose run --rm logprob python 02_bloco1_desempenho.py --smoke --limite 3 --dmax 10 --sem-bertscore
docker compose run --rm logprob python 03_bloco2_entropia.py --smoke --limite 3 --sem-bertscore
docker compose run --rm logprob python 04_metricas.py

# embeddings das perguntas (seção 4.2)
docker compose run --rm logprob python 01_embeddings_perguntas.py

# piloto obrigatório (seção 4.6)
docker compose run --rm logprob python 05_piloto.py

# blocos 1 e 2, um modelo por vez (seções 4.3 e 4.4)
docker compose run --rm logprob python 02_bloco1_desempenho.py --modelos gpt-oss:20b
docker compose run --rm logprob python 03_bloco2_entropia.py --modelos gpt-oss:20b

# consolidação (seção 4.5)
docker compose run --rm logprob python 04_metricas.py
```

Para uma sessão interativa, com vários comandos no mesmo contêiner:

```bash
docker compose run --rm logprob bash
```

O atalho da seção 4.8 também funciona:

```bash
docker compose run --rm logprob bash rodar_tudo.sh
```

### 5.1 Rodadas longas

Um bloco leva de 8 a 18 horas por modelo. Para que a queda do terminal não
derrube a rodada, use `-d` e acompanhe pelo log:

```bash
docker compose run -d --name b1_gemma3 logprob python 02_bloco1_desempenho.py --modelos gemma3:27b
docker logs -f b1_gemma3
```

Se ainda assim a rodada morrer, o comportamento é o da seção 4.7 do
`README.md`: reexecute o mesmo comando e ela retoma da pergunta em que parou.

---

## 6. Problemas específicos do contêiner

| Sintoma | Causa | O que fazer |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit ausente ou não configurado | instale-o e rode `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `unsupported option: gpus` | Compose anterior à v2.30 | troque `gpus: all` pelo bloco `deploy:` comentado no `docker-compose.yml` |
| `variable is not set` ao subir | `.env` ausente ou incompleto | `cp .env.exemplo .env` e preencha os caminhos |
| arquivos em `saidas/` pertencem ao root | `UID`/`GID` não definidos no `.env` | preencha com `id -u` / `id -g` e refaça o `docker compose build` |
| `No space left on device` durante o download | volume de `DIR_HF_HOST` cheio | seção 3.3 do `README.md`, ou aponte `DIR_HF_HOST` para um disco maior |
| `DataLoader worker killed` / erro de memória compartilhada | `shm_size` insuficiente | aumente o valor no `docker-compose.yml` |
| `CUDA error: no kernel image is available` | driver do host antigo demais | atualize o driver NVIDIA para ≥ 580 |

Os demais sintomas estão na seção 6 do `README.md` e se comportam igual dentro
do contêiner.

---

## 7. Entrega

Idêntica à seção 7 do `README.md`: zipe a pasta apontada por
`DIR_SAIDAS_HOST`, incluindo os `metadata_*.json`.

Para o registro de ambiente da tese, vale anotar junto o identificador exato
da imagem usada:

```bash
docker image inspect logprob:1.0 --format '{{.Id}}'
```
