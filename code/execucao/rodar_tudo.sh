#!/usr/bin/env bash
# rodar_tudo.sh -- executa o pipeline completo em sequência (Linux/macOS).
#
# Cada script pula o que já está feito, então este arquivo é seguro para
# reexecutar depois de uma interrupção.
#
# Uso:  bash rodar_tudo.sh
set -u   # sem -e de propósito: se um modelo falhar, os outros devem continuar

echo "### 0. Verificando ambiente"
python verificar_ambiente.py || { echo "Ambiente com problema. Abortando."; exit 1; }

echo
echo "### 1. Embeddings das perguntas (uma vez)"
python 01_embeddings_perguntas.py

echo
echo "### 1a. Sonda de memoria (o modelo cabe na placa no pior caso?)"
python verificar_memoria.py --modelos "gpt-oss:20b" || {
  echo "O modelo nao cabe no prompt longo. Veja a secao 9.14 do README."
  exit 1
}

echo
echo "### 1b. PILOTO (gpt-oss, 5 perguntas) -- §G do documento de qualificação"
python 05_piloto.py || {
  echo "PILOTO FALHOU. Não libere a máquina para a rodada longa; avise o Murilo."
  exit 1
}

echo
echo "### 2. BLOCO 1 -- desempenho (6 modelos)"
for m in "gpt-oss:20b" \
         "mistral-small3.1:latest" \
         "gemma3:27b" \
         "MedAIBase/MedGemma1.0:27b-it-q8_0" \
         "gemma4:31b" \
         "qwen3.8:27b"; do
  echo "--- Bloco 1: $m"
  python 02_bloco1_desempenho.py --modelos "$m"
done

echo
echo "### 3. BLOCO 2 -- entropia (4 modelos)"
for m in "gpt-oss:20b" \
         "mistral-small3.1:latest" \
         "gemma3:27b" \
         "MedAIBase/MedGemma1.0:27b-it-q8_0"; do
  echo "--- Bloco 2: $m"
  python 03_bloco2_entropia.py --modelos "$m"
done

echo
echo "### 4. Consolidação"
python 04_metricas.py

echo
echo "Pronto. Zipe a pasta saidas/ e devolva para o Murilo."
