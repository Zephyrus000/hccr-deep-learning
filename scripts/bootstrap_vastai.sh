#!/usr/bin/env bash
# Prepare dependencies and delegate dataset stages. No training is launched.
set -euo pipefail

repo_url=https://github.com/Zephyrus000/hccr-deep-learning.git
repo_dir="$PWD/hccr-deep-learning"
if [[ -f "$PWD/pyproject.toml" ]]; then repo_dir="$PWD"; fi
revision=main
revision_set=0
execute=0
install_python=0
torch_version=2.7.1
vision_version=0.22.1
torch_index=https://download.pytorch.org/whl/cu128
while (($#)); do
    case "$1" in
        --execute) execute=1; shift ;;
        --install-python) install_python=1; shift ;;
        --repo-url) repo_url="$2"; shift 2 ;;
        --repo-dir) repo_dir="$2"; shift 2 ;;
        --revision) revision="$2"; revision_set=1; shift 2 ;;
        --torch-version) torch_version="$2"; shift 2 ;;
        --torchvision-version) vision_version="$2"; shift 2 ;;
        --torch-index-url) torch_index="$2"; shift 2 ;;
        --) shift; break ;;
        *) printf 'Unknown bootstrap setup option. Put preparation options after --.\n' >&2; exit 2 ;;
    esac
done
if (( ! execute )); then
    printf 'Plan only: clone if needed, select Python >=3.12, create .venv, verify CUDA Torch, install project[modern] and gdown, then prepare data.\n'
    printf 'Use --execute before -- to perform setup and selected preparation stages. No training is launched.\n'
    exit 0
fi
if [[ "$revision" == -* ]]; then printf 'Invalid revision.\n' >&2; exit 2; fi
if [[ ! -d "$repo_dir" ]]; then
    if ! git clone -- "$repo_url" "$repo_dir" >/dev/null 2>&1; then
        printf 'Repository clone failed. Check access and connection.\n' >&2; exit 2
    fi
    if ! git -C "$repo_dir" fetch origin "$revision" >/dev/null 2>&1 || ! git -C "$repo_dir" checkout --detach FETCH_HEAD >/dev/null 2>&1; then
        printf 'Requested repository revision could not be checked out.\n' >&2; exit 2
    fi
elif [[ ! -f "$repo_dir/pyproject.toml" ]]; then
    printf 'Existing repository directory does not contain pyproject.toml.\n' >&2
    exit 2
elif (( revision_set )); then
    if [[ -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" ]]; then
        printf 'Existing checkout has local changes; use its current revision or a separate clone directory.\n' >&2; exit 2
    fi
    requested_commit="$(git -C "$repo_dir" rev-parse --verify --end-of-options "$revision^{commit}" 2>/dev/null)" || {
        printf 'Requested revision is not available in the existing checkout.\n' >&2; exit 2;
    }
    actual_commit="$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null)"
    if [[ "$requested_commit" != "$actual_commit" ]]; then
        printf 'Existing checkout differs from requested revision; use a separate clone directory.\n' >&2; exit 2
    fi
fi
cd "$repo_dir"
python_bin=
for candidate in python3.12 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info >= (3, 12); exec("import torch; assert torch.cuda.is_available()" if sys.version_info >= (3, 14) else "")' >/dev/null 2>&1; then
        python_bin="$(command -v "$candidate")"
        break
    fi
done
if [[ -z "$python_bin" ]] && (( install_python )); then
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'Install a system Python with pip first; managed Python setup needs python3.\n' >&2
        exit 2
    fi
    python3 -m pip install --user uv
    python3 -m uv python install 3.12
    python_bin="$(python3 -m uv python find 3.12)"
fi
if [[ -z "$python_bin" ]]; then
    printf 'Python >=3.12 is required. Supply it or explicitly use --install-python.\n' >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv --system-site-packages .venv
fi
.venv/bin/python -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'
.venv/bin/python -m pip install --upgrade pip
probe_torch() {
    .venv/bin/python - <<'PY'
import torch
import torchvision
assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 6)
assert torch.cuda.is_available() and torch.version.cuda
model = torch.nn.Conv2d(1, 2, 3).cuda()
image = torch.randn(2, 1, 8, 8, device="cuda", requires_grad=True)
model(image).square().mean().backward()
if torch.cuda.is_bf16_supported(including_emulation=False):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(image).square().mean()
    loss.backward()
torch.cuda.synchronize()
PY
}
if ! probe_torch >/dev/null 2>&1; then
    if ! .venv/bin/python -c 'import sys; assert (3, 12) <= sys.version_info < (3, 14)' >/dev/null 2>&1; then
        printf 'Pinned Torch fallback needs Python3.12/3.13. Select a compatible interpreter or use --install-python with a fresh .venv.\n' >&2; exit 2
    fi
    .venv/bin/python -m pip install "torch==$torch_version" "torchvision==$vision_version" --index-url "$torch_index"
fi
.venv/bin/python -m pip install -e '.[modern]' gdown
if ! probe_torch >/dev/null 2>&1; then
    printf 'CUDA convolution/backward runtime probe failed. Inspect GPU architecture, driver and Torch build.\n' >&2; exit 2
fi
printf 'CUDA convolution/backward runtime verified.\n'
exec .venv/bin/python scripts/bootstrap_vast.py --execute --root "$PWD" "$@"
