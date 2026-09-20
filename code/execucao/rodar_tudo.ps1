# rodar_tudo.ps1 -- executa o pipeline completo em sequência (Windows).
#
# Cada script pula o que já está feito, então este arquivo é seguro para
# reexecutar depois de uma interrupção.
#
# Uso:  .\rodar_tudo.ps1
# Se o PowerShell bloquear a execução:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Continue"   # um modelo que falha não derruba os outros

Write-Host "### 0. Verificando ambiente" -ForegroundColor Cyan
python verificar_ambiente.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ambiente com problema. Abortando." -ForegroundColor Red
    exit 1
}

Write-Host "`n### 1. Embeddings das perguntas (uma vez)" -ForegroundColor Cyan
python 01_embeddings_perguntas.py

Write-Host "`n### 1a. Sonda de memoria" -ForegroundColor Cyan
python verificar_memoria.py --modelos "gpt-oss:20b"
if ($LASTEXITCODE -ne 0) {
    Write-Host "O modelo nao cabe no prompt longo. Veja a secao 9.14 do README." -ForegroundColor Red
    exit 1
}

Write-Host "`n### 1b. PILOTO (gpt-oss, 5 perguntas)" -ForegroundColor Cyan
python 05_piloto.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "PILOTO FALHOU. Nao libere a maquina; avise o Murilo." -ForegroundColor Red
    exit 1
}

Write-Host "`n### 2. BLOCO 1 -- desempenho (6 modelos)" -ForegroundColor Cyan
$bloco1 = @(
    "gpt-oss:20b",
    "mistral-small3.1:latest",
    "gemma3:27b",
    "MedAIBase/MedGemma1.0:27b-it-q8_0",
    "gemma4:31b",
    "qwen3.8:27b"
)
foreach ($m in $bloco1) {
    Write-Host "--- Bloco 1: $m" -ForegroundColor Yellow
    python 02_bloco1_desempenho.py --modelos $m
}

Write-Host "`n### 3. BLOCO 2 -- entropia (4 modelos)" -ForegroundColor Cyan
$bloco2 = @(
    "gpt-oss:20b",
    "mistral-small3.1:latest",
    "gemma3:27b",
    "MedAIBase/MedGemma1.0:27b-it-q8_0"
)
foreach ($m in $bloco2) {
    Write-Host "--- Bloco 2: $m" -ForegroundColor Yellow
    python 03_bloco2_entropia.py --modelos $m
}

Write-Host "`n### 4. Consolidação" -ForegroundColor Cyan
python 04_metricas.py

Write-Host "`nPronto. Zipe a pasta saidas/ e devolva para o Murilo." -ForegroundColor Green
