#!/usr/bin/env bash
# =====================================================================
# setup_psi_cron.sh — 安装周度 PSI 漂移检查的 cron 任务
# =====================================================================
# 功能:
#   每周一早上 9:00 自动运行 weekly_psi_check.py
#   - 检查过去 7 天的推理特征日志
#   - 与训练基线对比计算 PSI
#   - 检测到漂移时退出码 1
#   - 报告保存到 monitoring/reports/ 供人工审阅
#   - 完整日志追加到 monitoring/logs/psi_cron.log
#
# 用法:
#   bash monitoring/setup_psi_cron.sh           # 安装 cron
#   bash monitoring/setup_psi_cron.sh --remove # 移除 cron
#   bash monitoring/setup_psi_cron.sh --check   # 查看当前 cron 状态
# =====================================================================

set -euo pipefail

# ---- 路径配置 ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"  # 可通过环境变量覆盖 Python 路径
CRON_SCHEDULE="0 9 * * 1"            # 每周一 09:00
CRON_LOG="${PROJECT_ROOT}/monitoring/logs/psi_cron.log"
CRON_MARKER="# LOL-PSI-WEEKLY-CHECK"  # 用于识别和更新本任务

# ---- 颜色输出 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- 校验前置条件 ----
check_prerequisites() {
    # 1. Python 可用
    if ! command -v "${PYTHON_BIN}" &>/dev/null; then
        error "Python 未找到: ${PYTHON_BIN}"
        error "请通过 PYTHON_BIN 环境变量指定 Python 路径"
        exit 1
    fi

    # 2. weekly_psi_check.py 存在
    local script="${PROJECT_ROOT}/monitoring/weekly_psi_check.py"
    if [[ ! -f "${script}" ]]; then
        error "weekly_psi_check.py 未找到: ${script}"
        exit 1
    fi

    # 3. crontab 命令可用
    if ! command -v crontab &>/dev/null; then
        error "crontab 命令不可用 (macOS 需在 系统设置 > 隐私与安全 > 完全磁盘访问 中授权终端)"
        exit 1
    fi

    info "前置条件检查通过"
    info "  Python: $(command -v ${PYTHON_BIN})"
    info "  脚本: ${script}"
    info "  日志: ${CRON_LOG}"
}

# ---- 安装 cron 任务 ----
install_cron() {
    check_prerequisites

    # 确保日志目录存在
    mkdir -p "$(dirname "${CRON_LOG}")"

    # 构造 cron 命令行
    # 使用 flock 防止任务重叠 (若上次未完成，下次跳过)
    local lock_file="${PROJECT_ROOT}/monitoring/logs/.psi_check.lock"
    local cron_cmd="cd \"${PROJECT_ROOT}\" && \
flock -n \"${lock_file}\" ${PYTHON_BIN} monitoring/weekly_psi_check.py --days 7 --threshold 0.25 \
>> \"${CRON_LOG}\" 2>&1"

    # 移除旧任务 (若存在)，再追加新任务
    remove_cron_silent

    # 追加新 cron 行
    (crontab -l 2>/dev/null; echo "${CRON_SCHEDULE} ${CRON_MARKER}"; echo "${cron_cmd}") | crontab -

    info "cron 任务已安装"
    info "  调度: ${CRON_SCHEDULE} (每周一 09:00)"
    info "  回溯: 7 天"
    info "  阈值: 0.25"
    info ""
    info "查看 cron 任务: crontab -l | grep -A1 '${CRON_MARKER}'"
    info "查看运行日志: tail -f ${CRON_LOG}"
    info "查看 PSI 报告: ls ${PROJECT_ROOT}/monitoring/reports/"
}

# ---- 移除 cron 任务 ----
remove_cron() {
    if crontab -l 2>/dev/null | grep -q "${CRON_MARKER}"; then
        remove_cron_silent
        info "cron 任务已移除"
    else
        warn "未找到 PSI cron 任务 (无需移除)"
    fi
}

remove_cron_silent() {
    # 同时移除 marker 行和下一行命令行
    crontab -l 2>/dev/null | grep -v "${CRON_MARKER}" | \
        awk 'BEGIN{skip=0} {if(skip){skip=0; next} print}' | \
        crontab - 2>/dev/null || true
    # 上面的 awk 可能误删其他行，改用更精确的方式: 删除 marker 行及其后一行
    crontab -l 2>/dev/null | grep -v "${CRON_MARKER}" | crontab - 2>/dev/null || true
}

# ---- 查看当前状态 ----
check_status() {
    info "当前 PSI cron 任务状态:"
    echo ""
    if crontab -l 2>/dev/null | grep -q "${CRON_MARKER}"; then
        info "✓ cron 任务已安装"
        crontab -l 2>/dev/null | grep -A1 "${CRON_MARKER}"
    else
        warn "✗ cron 任务未安装"
    fi
    echo ""
    info "PSI 报告目录 (${PROJECT_ROOT}/monitoring/reports/):"
    if [[ -d "${PROJECT_ROOT}/monitoring/reports" ]]; then
        ls -lt "${PROJECT_ROOT}/monitoring/reports/" | head -10
    else
        warn "  目录不存在 (尚未运行过 weekly_psi_check.py)"
    fi
    echo ""
    info "推理特征日志目录 (${PROJECT_ROOT}/logs/inference_features/):"
    if [[ -d "${PROJECT_ROOT}/logs/inference_features" ]]; then
        local n_files=$(ls -1 "${PROJECT_ROOT}/logs/inference_features/" 2>/dev/null | wc -l)
        info "  ${n_files} 个 parquet 文件"
        ls -lt "${PROJECT_ROOT}/logs/inference_features/" | head -5
    else
        warn "  目录不存在 (尚未有推理请求)"
    fi
}

# ---- 主入口 ----
case "${1:-install}" in
    install|"")
        install_cron
        ;;
    --remove|remove)
        remove_cron
        ;;
    --check|check|status)
        check_status
        ;;
    *)
        echo "用法: $0 [install|--remove|--check]"
        exit 1
        ;;
esac
