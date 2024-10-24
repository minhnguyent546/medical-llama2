#!/usr/bin/env bash

if command -v nvcc &>/dev/null; then
  nvcc --version
  exit 0
fi

if ! command -v gcc &>/dev/null; then
  apt update
  apt install build-essential
fi

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

nvcc --version
cat > ~/.bashrc << EOF
export PATH="${PATH}:/usr/local/cuda-12.6/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-12.6/lib64/"
EOF

echo "Please refresh your shell with the command below:"
echo "source ~/.bashrc"
