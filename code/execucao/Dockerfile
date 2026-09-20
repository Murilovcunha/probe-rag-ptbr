# ============================================================
# Dockerfile -- experimentos RAG iterativo/incremental + entropia
#
# Base: imagem oficial do PyTorch com CUDA 13.2, que traz kernels
# Blackwell (sm_120). Isto resolve o problema descrito no
# requirements.txt: nao instale torch pelo PyPI generico aqui.
#
# Alternativa validada, caso o piloto (05_piloto.py) falhe nesta:
#   pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
# Trocar a base EXIGE refazer o piloto e anotar o novo sha256 da imagem.
#
# Build:
#   docker compose build
# ============================================================
FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf \
    LOGPROB_DADOS=/app/dados \
    LOGPROB_SAIDAS=/saidas

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencias primeiro, para aproveitar o cache de camadas.
# --break-system-packages: a imagem base usa o Python do sistema (PEP 668).
# O torch ja vem instalado nela e NAO deve ser reinstalado pelo PyPI.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Trava: se alguma dependencia tiver arrastado um torch generico do PyPI por
# cima do build CUDA da imagem base, o build falha AQUI -- e nao seis horas
# depois, com "no kernel image is available for execution on the device".
RUN python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("arquiteturas:", torch.cuda.get_arch_list())
assert torch.version.cuda, "torch sem CUDA: foi substituido por um build CPU"
assert "sm_120" in torch.cuda.get_arch_list(), \
    "torch sem kernels sm_120 (Blackwell): base trocada ou torch reinstalado"
PY

# Usuario nao-root com o mesmo UID/GID do host: sem isso, tudo que o
# contêiner escreve em /hf e /saidas fica pertencendo ao root la fora.
ARG UID=1000
ARG GID=1000
RUN if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" exp; fi \
 && if ! getent passwd "${UID}" >/dev/null; then \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash exp; \
    fi \
 && mkdir -p /hf /saidas \
 && chown -R "${UID}:${GID}" /hf /saidas /app

# codigo por ultimo: muda com mais frequencia que as dependencias.
# Em uso normal o docker-compose monta a pasta do host por cima de /app;
# esta copia existe para a imagem ser autossuficiente (arquivo de replicacao).
COPY --chown=${UID}:${GID} . /app

USER ${UID}:${GID}

CMD ["bash"]
