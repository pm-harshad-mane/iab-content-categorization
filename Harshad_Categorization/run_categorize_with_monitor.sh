#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <output_jsonl> <monitor_log> <stdout_log> [categorize args...]" >&2
  exit 2
fi

output_jsonl="$1"
monitor_log="$2"
stdout_log="$3"
shift 3

sample_interval="${SAMPLE_INTERVAL:-0.2}"

python3 categorize_on_hosted_models.py --output-jsonl "$output_jsonl" "$@" > "$stdout_log" 2>&1 &
job_pid=$!

{
  echo "job_pid=${job_pid}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "sample_interval_seconds=${sample_interval}"
  echo "stdout_log=${stdout_log}"
  echo
} > "$monitor_log"

while kill -0 "$job_pid" 2>/dev/null; do
  {
    echo "ts=$(date '+%Y-%m-%d %H:%M:%S')"
    ps -p "$job_pid" -o pid=,ppid=,%cpu=,%mem=,rss=,vsz=,etime=,cmd=
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo "---"
  } >> "$monitor_log"
  sleep "$sample_interval"
done

wait "$job_pid"
exit_code=$?

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "exit_code=${exit_code}"
} >> "$monitor_log"

exit "$exit_code"
