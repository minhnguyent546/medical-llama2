#!/usr/bin/env bash

set -e

if command -v nvcc &>/dev/null; then
  nvcc --version
  exit 0
fi

apt update
apt install -y wget

if ! command -v gcc &>/dev/null; then
  apt install -y build-essential
fi

if [[ ! -f 'cuda-ubuntu2204.pin' ]]; then
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O cuda-ubuntu2204.pin
fi

mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

if [[ ! -f 'cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb' ]]; then
  wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb -O cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
fi

dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-6

/usr/local/cuda-12.6/bin/nvcc --version
cat > ~/.bashrc << EOF
export PATH="${PATH}:/usr/local/cuda-12.6/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-12.6/lib64/"
EOF

echo "Please refresh your shell with the following command: source ~/.bashrc"
