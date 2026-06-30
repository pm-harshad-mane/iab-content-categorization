#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FILES=(
  "02_fetched_url_content_files/HarshardData_New_1M_01.jsonl"
  "02_fetched_url_content_files/HarshardData_New_1M_02.jsonl"
  "02_fetched_url_content_files/HarshardData_New_1M_03.jsonl"
  "02_fetched_url_content_files/HarshardData_New_1M_04.jsonl"
  "02_fetched_url_content_files/HarshardData_New_1M_05.jsonl"
  "02_fetched_url_content_files/HarshardData_New_1M_06.jsonl"
)

API_BASE="http://127.0.0.1:8000"
WORKERS="1000"
GPU_ID="${GPU_ID:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVER_PID=""

wait_for_server() {
  local attempts=0
  until curl -sS --max-time 5 "${API_BASE}/v1/models" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge 120 ]]; then
      echo "Server did not become ready on ${API_BASE}" >&2
      return 1
    fi
    sleep 5
  done
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

start_server() {
  local model="$1"
  local server_log="$2"
  stop_server
  echo "Starting vLLM for ${model}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup python3 -u -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --runner pooling \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    >"${server_log}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server
}

run_embeddings_for_model() {
  local model="$1"
  local server_log="$2"
  start_server "${model}" "${server_log}"
  for input_file in "${FILES[@]}"; do
    echo "Embedding ${input_file} with ${model}"
    python3 content-fetching/02_generate_url_embeddings.py \
      --model "${model}" \
      --api-base "${API_BASE}" \
      --workers "${WORKERS}" \
      --input-files "${input_file}"
  done
}

trap stop_server EXIT

run_embeddings_for_model "BAAI/bge-m3" "03_fetched_url_content_embedding_files/bge_m3_vllm.nohup.log"
run_embeddings_for_model "google/embeddinggemma-300m" "03_fetched_url_content_embedding_files/embeddinggemma_300m_vllm.nohup.log"

echo "All missing stage-03 embedding jobs completed."
