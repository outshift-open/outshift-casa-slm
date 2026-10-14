#!/usr/bin/env bash
# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/pixi_runtime.sh"

# ---------------------------------------------------------------------------
# Run ASTRA data generation experiments across tool-count and split combos.
#
# Usage:
#   bash run_experiments.sh                              # ASTRA test, 1+2+3 tool
#   bash run_experiments.sh --tools "1 3"                 # 1-tool and 3-tool only
#   bash run_experiments.sh --split validation            # ASTRA validation split
#   bash run_experiments.sh --seed 42                     # override seed
#
# Agentic generation uses ASTRA task data and MCP tool catalogs.
# ---------------------------------------------------------------------------

SEED=13
MAX_ATTEMPTS=20
MODEL="azure/gpt-5.2"
OUT_ROOT="./generated_agentic_conversations"
DATA_ROOT="${REPOSITORY_ROOT}/data/conversation"
TASK_TOOL_ROOT="${DATA_ROOT}/task_tool/single_mcp_server"
MCP_SERVERS_DIR="${DATA_ROOT}/mcp_servers"
SPLIT="test"
TOOLS="1 2 3"
BENCHMARKS="ASTRA"
STALL_TIMEOUT=1800

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2";           shift 2 ;;
        --max-attempts)   MAX_ATTEMPTS="$2";   shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --out-root)       OUT_ROOT="$2";       shift 2 ;;
        --tools)          TOOLS="$2";          shift 2 ;;
        --benchmarks)     BENCHMARKS="$2";     shift 2 ;;
        --split)          SPLIT="$2";          shift 2 ;;
        --stall-timeout)  STALL_TIMEOUT="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$SPLIT" != "test" && "$SPLIT" != "validation" ]]; then
    echo "Invalid split: $SPLIT (must be test or validation)" >&2
    exit 1
fi

for bench in $BENCHMARKS; do
    if [[ "$bench" != "ASTRA" ]]; then
        echo "Invalid benchmark: $bench (Agentic generation is ASTRA-only)" >&2
        exit 1
    fi
done

TOOL_DIR_MAP=([1]="01_one_tool" [2]="02_two_tools" [3]="03_three_tools")
LABEL_DIR_MAP=([1]="01_tool" [2]="02_tools" [3]="03_tools")

EXPERIMENTS=()
for t in $TOOLS; do
    dir="${TOOL_DIR_MAP[$t]:-}"
    if [[ -z "$dir" ]]; then
        echo "Invalid tool count: $t (must be 1, 2, or 3)" >&2
        exit 1
    fi
    for bench in $BENCHMARKS; do
        label="${LABEL_DIR_MAP[$t]}"
        data_file="${TASK_TOOL_ROOT}/${dir}/${bench}/03_grouped_filtered/${SPLIT}_curated_grouped.json"
        if [[ ! -f "$data_file" ]]; then
            echo "Data file not found: $data_file" >&2
            exit 1
        fi
        EXPERIMENTS+=("${data_file}:${label}_${bench}:${bench}")
    done
done

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

count_dialogs() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -name '*.json' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

WATCHDOG_PID=""

cleanup() {
    if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    exit 130
}

trap cleanup INT TERM

run_with_watchdog() {
    local out_dir="$1"
    shift
    local dialogs_dir="${out_dir}/dialogs"

    while true; do
        local last_count
        last_count=$(count_dialogs "$dialogs_dir")
        local last_progress_time
        last_progress_time=$(date +%s)

        "$@" &
        local pid=$!
        WATCHDOG_PID=$pid

        local exited=false
        while true; do
            sleep 60

            if ! kill -0 "$pid" 2>/dev/null; then
                local exit_code=0
                wait "$pid" || exit_code=$?
                WATCHDOG_PID=""
                exited=true
                if [[ $exit_code -eq 0 ]]; then
                    return 0
                else
                    echo "  [watchdog] Process exited with code $exit_code, restarting..." >&2
                    break
                fi
            fi

            local current_count
            current_count=$(count_dialogs "$dialogs_dir")
            if [[ "$current_count" -gt "$last_count" ]]; then
                last_count=$current_count
                last_progress_time=$(date +%s)
            fi

            local now
            now=$(date +%s)
            local elapsed=$(( now - last_progress_time ))
            if [[ $elapsed -ge $STALL_TIMEOUT ]]; then
                echo "  [watchdog] No progress for ${STALL_TIMEOUT}s (stuck at $last_count dialogs). Killing pid $pid..." >&2
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
                WATCHDOG_PID=""
                echo "  [watchdog] Restarting..." >&2
                break
            fi
        done

        if [[ "$exited" == "true" ]]; then
            continue
        fi
    done
}

for entry in "${EXPERIMENTS[@]}"; do
    DATA_FILE="${entry%%:*}"
    BENCH="${entry##*:}"
    LABEL="${entry#*:}"
    LABEL="${LABEL%:*}"
    CURRENT=$((CURRENT + 1))

    OUT_DIR="${OUT_ROOT}/${SPLIT}/${LABEL}"
    BENCH_MCP_DIR="${MCP_SERVERS_DIR}/${BENCH}"

    echo ""
    echo "================================================================"
    echo "  [$CURRENT/$TOTAL] $LABEL"
    echo "  split:  $SPLIT"
    echo "  data:   $DATA_FILE"
    echo "  mcp:    $BENCH_MCP_DIR"
    echo "  output: $OUT_DIR"
    echo "================================================================"
    echo ""

    run_with_watchdog "$OUT_DIR" \
        "$PYTHON_BIN" generate_dataset.py \
            --out "$OUT_DIR" \
            --seed "$SEED" \
            --max-attempts-per-example "$MAX_ATTEMPTS" \
            --model "$MODEL" \
            --task-data "$DATA_FILE" \
            --mcp-servers-dir "$BENCH_MCP_DIR" \
            --resume
done

echo ""
echo "All $TOTAL ${SPLIT} experiments complete. Results in $OUT_ROOT/$SPLIT/"
