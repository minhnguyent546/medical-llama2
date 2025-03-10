#!/usr/bin/env bash

# this script is used to set up for fine-tuning on fresh remote machine
# run this script via: `source remote_machine_setup.sh`

export HF_TOKEN='<YOUR_HF_TOKEN>'
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'
export HF_HUB_ENABLE_HF_TRANSFER=1 # for faster downloading

DEFAULT_BRANCH="master"
PROJECT_DIR_NAME="medical-llama2"

apt update
apt install -y git tmux neofetch htop speedtest-cli tree time nano wget net-tools python3-pip python3-venv

if [[ ! -d .venv ]]; then
  echo ".venv not found, creating one"
  python3 -m venv .venv
fi

source .venv/bin/activate

if [[ ! -d "$PROJECT_DIR_NAME" ]]; then
  git clone --branch "$DEFAULT_BRANCH" "https://github.com/minhnguyent546/medical-llama2.git" "$PROJECT_DIR_NAME"
  pip install -r "${PROJECT_DIR_NAME}/requirements.txt"
fi
