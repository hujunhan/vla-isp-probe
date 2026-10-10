# Shared paths for the shell scripts. Override any of these before running. [ai-scaffold]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENVLA_OFT_ROOT:=$HOME/research/openvla-oft}"
: "${LIBERO_PLUS_ROOT:=$HOME/research/LIBERO-plus}"
: "${INFINITE_ISP_ROOT:=$HOME/research/Infinite-ISP}"
: "${LOG_DIR:=$HOME/research/logs}"
: "${CONDA_SH:=$HOME/miniconda3/etc/profile.d/conda.sh}"
export REPO_ROOT OPENVLA_OFT_ROOT LIBERO_PLUS_ROOT INFINITE_ISP_ROOT LOG_DIR
mkdir -p "$LOG_DIR"
