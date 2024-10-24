#!/usr/bin/env bash

# run this script with: `source vastai_setup.sh`

export HF_TOKEN='<YOUR_HF_ACCESS_TOKEN>'
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'

DEFAULT_BRANCH="dev"
PROJECT_DIR_NAME="medical-llama2"

apt update
apt install -y neofetch htop speedtest-cli tree time

if [[ ! -d .venv ]]; then
  echo ".venv not found, creating one"
  python3 -m venv .venv
fi

source .venv/bin/activate

if [[ ! -d "$PROJECT_DIR_NAME" ]]; then
  git clone --branch "$DEFAULT_BRANCH" "https://github.com/minhnguyent546/medical-llama2.git" "$PROJECT_DIR_NAME"
  pip install -r "${PROJECT_DIR_NAME}/requirements.txt"
fi
