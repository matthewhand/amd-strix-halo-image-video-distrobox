#!/bin/bash
set -e

# Install Zabbly kernel (6.18+) and ROCm on Ubuntu for Strix Halo.
# Run on the HOST, not inside the container.
#
# Usage:
#   sudo ./install_host.sh              # install both kernel + ROCm
#   sudo ./install_host.sh --kernel     # kernel only
#   sudo ./install_host.sh --rocm       # ROCm only

INSTALL_KERNEL=true
INSTALL_ROCM=true

case "${1:-}" in
    --kernel) INSTALL_ROCM=false ;;
    --rocm)   INSTALL_KERNEL=false ;;
esac

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo $0 $*"
    exit 1
fi

apt-get update
apt-get install -y wget ca-certificates gnupg2

# --- Zabbly Kernel ---
if $INSTALL_KERNEL; then
    echo "Installing Zabbly kernel (6.18+)..."
    mkdir -p /etc/apt/keyrings
    wget -qO - https://pkgs.zabbly.com/key.asc | tee /etc/apt/keyrings/zabbly.asc > /dev/null

    cat > /etc/apt/sources.list.d/zabbly-kernel-stable.sources <<'EOF'
Enabled: yes
Types: deb
URIs: https://pkgs.zabbly.com/kernel/stable
Suites: noble
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/zabbly.asc
EOF

    apt-get update
    apt-get install -y linux-zabbly
    echo "Kernel installed. Reboot required."
fi

# --- ROCm ---
if $INSTALL_ROCM; then
    echo "Installing ROCm..."
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg --yes
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2/ noble main" > /etc/apt/sources.list.d/rocm.list
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/7.2/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600

    apt-get update
    apt-get install -y rocm-hip-libraries rocm-hip-runtime rocm-core rocm-smi-lib
    usermod -aG render,video "${SUDO_USER:-$USER}"
    echo "ROCm installed. Log out and back in for group changes."
fi

echo "Done. Reboot to activate new kernel."
