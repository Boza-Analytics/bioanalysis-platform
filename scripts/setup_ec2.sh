#!/usr/bin/env bash
#
# Fully unattended provisioner for the BioAnalysis platform on a fresh
# Ubuntu 22.04 EC2 instance. Run as the `ubuntu` user:
#
#     curl -fsSL <raw-url>/setup_ec2.sh | sudo -E -u ubuntu bash
#   or, with the repo already on the box:
#     bash scripts/setup_ec2.sh
#
# Configuration via environment variables (all optional):
#   REPO_URL   git URL to clone if the project is not already present
#   HF_TOKEN   Hugging Face token for the gated weights (lal-Joey/MedSAM3_v1)
#
# The script is idempotent-ish and never prompts.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
# Stop apt/needrestart from popping interactive dialogs.
export NEEDRESTART_MODE=a

PROJECT_DIR="/home/ubuntu/bioanalysis-platform"
VENV_DIR="${PROJECT_DIR}/.venv"
WEIGHTS_DIR="${PROJECT_DIR}/submodules/MedSAM3/weights/medsam3_v1"
REPO_URL="${REPO_URL:-}"
HF_TOKEN="${HF_TOKEN:-}"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }

# --- 1. Base system --------------------------------------------------------
log "Updating base system"
sudo -E apt-get update -y
sudo -E apt-get upgrade -y

# --- 2. Core packages ------------------------------------------------------
log "Installing core packages (Python 3.11, git, docker, nginx)"
sudo -E apt-get install -y software-properties-common curl ca-certificates gnupg lsb-release
sudo -E add-apt-repository -y ppa:deadsnakes/ppa
sudo -E apt-get update -y
sudo -E apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git nginx \
    docker.io docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu || true

# --- 3. NVIDIA driver + CUDA 12.1 toolkit ----------------------------------
# Skipped automatically on non-GPU hosts (no NVIDIA PCI device present).
# NOTE: torch's pip wheels (cu12x) bundle their own CUDA runtime, so all we
# truly need from the host is a WORKING NVIDIA DRIVER.
if lspci 2>/dev/null | grep -qi nvidia; then
    if nvidia-smi >/dev/null 2>&1; then
        log "NVIDIA driver already working — skipping driver install"
    else
        log "Installing NVIDIA driver + CUDA 12.1 toolkit"
        # The AWS HWE kernel (6.8-aws) is built with gcc-12. Ubuntu 22.04 defaults
        # to gcc-11, which lacks '-ftrivial-auto-var-init=zero' and makes the
        # NVIDIA dkms module build FAIL. Install gcc-12 and make it the default
        # BEFORE the driver so the kernel module compiles cleanly.
        sudo -E apt-get install -y gcc-12 "linux-headers-$(uname -r)"
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60
        sudo update-alternatives --set gcc /usr/bin/gcc-12

        KEYRING=/usr/share/keyrings/cuda-archive-keyring.gpg
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
            | sudo gpg --dearmor -o "${KEYRING}" || true
        echo "deb [signed-by=${KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
            | sudo tee /etc/apt/sources.list.d/cuda.list >/dev/null
        sudo -E apt-get update -y
        sudo -E apt-get install -y cuda-toolkit-12-1 nvidia-driver-535 || \
            sudo -E apt-get install -y nvidia-cuda-toolkit
        # Ensure the dkms module is actually built/installed (idempotent).
        sudo dpkg --configure -a || true
        sudo modprobe nvidia 2>/dev/null || \
            log "nvidia module not loaded yet — a reboot may be required before first use"
    fi
else
    log "No NVIDIA GPU detected — skipping CUDA install (CPU inference only)"
fi

# --- 4. Project source -----------------------------------------------------
if [ ! -d "${PROJECT_DIR}" ]; then
    if [ -n "${REPO_URL}" ]; then
        log "Cloning project from ${REPO_URL}"
        git clone "${REPO_URL}" "${PROJECT_DIR}"
    else
        echo "ERROR: ${PROJECT_DIR} missing and REPO_URL not set." >&2
        exit 1
    fi
else
    log "Project already present at ${PROJECT_DIR}"
fi
cd "${PROJECT_DIR}"

# Ensure the two source submodules are present.
if [ ! -d submodules/MedSAM3/.git ] && [ ! -f submodules/MedSAM3/setup.py ]; then
    log "Cloning MedSAM3"
    git clone https://github.com/Joey-S-Liu/MedSAM3 submodules/MedSAM3
fi
if [ ! -d submodules/SynthMT/.git ] && [ ! -f submodules/SynthMT/pyproject.toml ]; then
    log "Cloning SynthMT"
    git clone https://github.com/ml-lab-htw/SynthMT submodules/SynthMT
fi

# --- 5-7. Python environment ----------------------------------------------
log "Creating virtualenv and installing Python dependencies"
python3.11 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel "huggingface_hub[cli]"

pip install -e submodules/MedSAM3
pip install -e submodules/SynthMT || echo "WARN: SynthMT editable install failed (non-fatal for MVP)"
pip install -r requirements.txt

# --- 8. Download MedSAM3 weights (gated) -----------------------------------
mkdir -p "${WEIGHTS_DIR}"
if [ -f "${WEIGHTS_DIR}/best_lora_weights.pt" ]; then
    log "Weights already present — skipping download"
elif [ -n "${HF_TOKEN}" ]; then
    log "Downloading MedSAM3 weights from Hugging Face"
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
    huggingface-cli download lal-Joey/MedSAM3_v1 --local-dir "${WEIGHTS_DIR}"
else
    echo "WARN: HF_TOKEN not set — skipping gated weight download." >&2
    echo "      The API will boot but /analyse returns 503 until weights exist at:" >&2
    echo "      ${WEIGHTS_DIR}/best_lora_weights.pt" >&2
fi
deactivate

# --- 9. nginx reverse proxy ------------------------------------------------
log "Configuring nginx (/ -> frontend, /api -> uvicorn :8000)"
sudo tee /etc/nginx/sites-available/bioanalysis >/dev/null <<NGINX
server {
    listen 80 default_server;
    server_name _;
    client_max_body_size 64M;

    root ${PROJECT_DIR}/frontend;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/bioanalysis /etc/nginx/sites-enabled/bioanalysis
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# --- 10. systemd service ---------------------------------------------------
log "Installing systemd service: bioanalysis.service"
sudo tee /etc/systemd/system/bioanalysis.service >/dev/null <<SERVICE
[Unit]
Description=BioAnalysis FastAPI server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_DIR}/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE
sudo systemctl daemon-reload
sudo systemctl enable bioanalysis.service
sudo systemctl restart bioanalysis.service

# --- 11. Summary -----------------------------------------------------------
log "Verifying services"
sleep 3
systemctl is-active --quiet bioanalysis.service && echo "bioanalysis.service: active" || echo "bioanalysis.service: NOT active"
systemctl is-active --quiet nginx && echo "nginx: active" || echo "nginx: NOT active"

# Fetch public IP via IMDSv2.
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" \
    http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<unknown>")

cat <<DONE

============================================================
 BioAnalysis is provisioned.
   Frontend (nginx) : http://${PUBLIC_IP}/
   API (direct)     : http://${PUBLIC_IP}:8000/health
   API (via nginx)  : http://${PUBLIC_IP}/api/health
 Logs: sudo journalctl -u bioanalysis -f
============================================================
DONE
