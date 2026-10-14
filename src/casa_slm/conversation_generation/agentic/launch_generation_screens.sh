#!/usr/bin/env bash
# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/pixi_runtime.sh"

# Launch ASTRA curated generation jobs in detached screen sessions.
#
# Usage:
#   bash launch_generation_screens.sh single_mcp_server
#   bash launch_generation_screens.sh cross_mcp_server
#   bash launch_generation_screens.sh single_mcp_server --dry-run

SERVER_TYPE="${1:-}"
if [[ -z "$SERVER_TYPE" ]]; then
    echo "Usage: bash launch_generation_screens.sh <single_mcp_server|cross_mcp_server> [options]" >&2
    exit 1
fi
shift

SEED=13
MAX_ATTEMPTS=20
MODEL="azure/gpt-5.2"
OUT_ROOT="./generated_agentic_conversations"
DATA_ROOT="${REPOSITORY_ROOT}/data/conversation"
MCP_SERVERS_DIR="${DATA_ROOT}/mcp_servers"
SPLITS="test validation"
SINGLE_TOOLS="1 2 3"
CROSS_TOOLS="2 3"
DRY_RUN=0
STALL_TIMEOUT=1800

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2";           shift 2 ;;
        --max-attempts)   MAX_ATTEMPTS="$2";   shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --out-root)       OUT_ROOT="$2";       shift 2 ;;
        --data-root)      DATA_ROOT="$2";      shift 2 ;;
        --stall-timeout)  STALL_TIMEOUT="$2";  shift 2 ;;
        --dry-run)        DRY_RUN=1;           shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

MCP_SERVERS_DIR="${DATA_ROOT}/mcp_servers"

case "$SERVER_TYPE" in
    single_mcp_server)
        TOOLS="$SINGLE_TOOLS"
        ;;
    cross_mcp_server)
        TOOLS="$CROSS_TOOLS"
        ;;
    *)
        echo "Invalid server type: $SERVER_TYPE (must be single_mcp_server or cross_mcp_server)" >&2
        exit 1
        ;;
esac

if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "screen is required but was not found on PATH" >&2
        exit 1
    fi
fi

tool_dir() {
    case "$1" in
        1) echo "01_one_tool" ;;
        2) echo "02_two_tools" ;;
        3) echo "03_three_tools" ;;
        *)
            echo "Invalid tool count: $1" >&2
            exit 1
            ;;
    esac
}

tool_label() {
    case "$1" in
        1) echo "01_tool" ;;
        2) echo "02_tools" ;;
        3) echo "03_tools" ;;
        *)
            echo "Invalid tool count: $1" >&2
            exit 1
            ;;
    esac
}

count_dialogs() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -name '*.json' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

write_screen_runner() {
    local runner_path="$1"
    local out_dir="$2"
    local data_file="$3"
    local mcp_dir="$4"
    local log_file="$5"

    local quoted_python
    printf -v quoted_python '%q' "$PYTHON_BIN"

    cat > "$runner_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$(pwd)"

out_dir="$out_dir"
data_file="$data_file"
mcp_dir="$mcp_dir"
log_file="$log_file"
python_bin=$quoted_python
pid=""
tail_pid=""


count_dialogs() {
    local dir="\$1"
    if [[ -d "\$dir" ]]; then
        find "\$dir" -maxdepth 1 -name '*.json' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

NON_RESTARTABLE_PATTERNS='Pool exhausted|Failed to generate a valid'

stop_log_tail() {
    if [[ -n "\${tail_pid:-}" ]] && kill -0 "\$tail_pid" 2>/dev/null; then
        kill "\$tail_pid" 2>/dev/null || true
        wait "\$tail_pid" 2>/dev/null || true
    fi
}

stop_child() {
    if [[ -n "\${pid:-}" ]] && kill -0 "\$pid" 2>/dev/null; then
        echo "  [runner] Stopping generation pid \$pid..." >> "\$log_file"
        kill "\$pid" 2>/dev/null || true
        wait "\$pid" 2>/dev/null || true
    fi
}

handle_interrupt() {
    echo "  [runner] Ctrl-C received; stopping generation." >> "\$log_file"
    stop_child
    stop_log_tail
    echo "  [runner] Generation stopped. Type 'exit' to close this screen, or press Ctrl-A then D to detach." | tee -a "\$log_file"
    exec "\${SHELL:-/bin/bash}" -l
}

handle_terminate() {
    echo "  [runner] Termination requested; stopping generation." >> "\$log_file"
    stop_child
    stop_log_tail
    exit 143
}

trap handle_interrupt INT
trap handle_terminate TERM HUP

touch "\$log_file"
if [[ -z "\${HTTPS_PROXY:-}\${https_proxy:-}\${HTTP_PROXY:-}\${http_proxy:-}" ]]; then
    echo "  [runner] Warning: no HTTP(S) proxy variables are set; outbound API calls may hang on restricted hosts." >> "\$log_file"
fi
tail -n 40 -F "\$log_file" &
tail_pid=\$!

while true; do
    last_count=\$(count_dialogs "\$out_dir/dialogs")
    last_progress_time=\$(date +%s)
    "\$python_bin" -u generate_dataset.py \\
        --out "\$out_dir" \\
        --seed "$SEED" \\
        --max-attempts-per-example "$MAX_ATTEMPTS" \\
        --model "$MODEL" \\
        --task-data "\$data_file" \\
        --mcp-servers-dir "\$mcp_dir" \\
        --resume >> "\$log_file" 2>&1 &
    pid=\$!

    while true; do
        sleep 60
        if ! kill -0 "\$pid" 2>/dev/null; then
            exit_code=0
            wait "\$pid" || exit_code=\$?
            if [[ \$exit_code -eq 0 ]]; then
                stop_log_tail
                exit 0
            fi
            if grep -E "\$NON_RESTARTABLE_PATTERNS" "\$log_file" >/dev/null 2>&1; then
                echo "  [watchdog] Non-restartable generation failure detected. Leaving screen runner stopped." >> "\$log_file"
                stop_log_tail
                exit "\$exit_code"
            fi
            echo "  [watchdog] Process exited with code \$exit_code, restarting..." >> "\$log_file"
            break
        fi

        current_count=\$(count_dialogs "\$out_dir/dialogs")
        if [[ "\$current_count" -gt "\$last_count" ]]; then
            last_count="\$current_count"
            last_progress_time=\$(date +%s)
        fi

        now=\$(date +%s)
        elapsed=\$(( now - last_progress_time ))
        if [[ \$elapsed -ge $STALL_TIMEOUT ]]; then
            echo "  [watchdog] No progress for ${STALL_TIMEOUT}s (stuck at \$last_count dialogs). Killing pid \$pid..." >> "\$log_file"
            kill "\$pid" 2>/dev/null || true
            wait "\$pid" 2>/dev/null || true
            echo "  [watchdog] Restarting..." >> "\$log_file"
            break
        fi
    done
done
EOF
    chmod +x "$runner_path"
}

launched=0
for SPLIT in $SPLITS; do
    for TOOL_COUNT in $TOOLS; do
        DIR="$(tool_dir "$TOOL_COUNT")"
        BASE_LABEL="$(tool_label "$TOOL_COUNT")"

        DATA_FILE="${DATA_ROOT}/task_tool/${SERVER_TYPE}/${DIR}/ASTRA/03_grouped_filtered/${SPLIT}_curated_grouped.json"
        MCP_DIR="${MCP_SERVERS_DIR}/ASTRA"
        OUT_DIR="${OUT_ROOT}/task_tool/${SERVER_TYPE}/${DIR}/ASTRA/03_grouped_filtered/${SPLIT}_curated_grouped"
        SCREEN_NAME="astra_${SERVER_TYPE}_${SPLIT}_${BASE_LABEL}"
        RUNNER_PATH="${OUT_DIR}/screen_runner.sh"
        LOG_FILE="${OUT_DIR}/run.log"

        if [[ ! -f "$DATA_FILE" ]]; then
            echo "Data file not found: $DATA_FILE" >&2
            exit 1
        fi

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "screen=$SCREEN_NAME"
            echo "data=$DATA_FILE"
            echo "output=$OUT_DIR"
            echo "log=$LOG_FILE"
            continue
        fi

        mkdir -p "$OUT_DIR"
        RUNNER_PATH="$(cd "$OUT_DIR" && pwd)/screen_runner.sh"
        write_screen_runner "$RUNNER_PATH" "$OUT_DIR" "$DATA_FILE" "$MCP_DIR" "$LOG_FILE"
        {
            echo ""
            echo "--- screen launch $(date) ---"
            echo "screen: $SCREEN_NAME"
            echo "data: $DATA_FILE"
            echo "output: $OUT_DIR"
        } >> "$LOG_FILE"

        # The detached login shell expands its own positional argument.
        # shellcheck disable=SC2016
        screen -dmS "$SCREEN_NAME" bash -lc 'exec "$1"' screen-runner "$RUNNER_PATH"
        echo "Launched $SCREEN_NAME -> $OUT_DIR"
        launched=$((launched + 1))
    done
done

if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Launched $launched screen session(s)."
fi
