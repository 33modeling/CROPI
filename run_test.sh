#!/bin/bash
export GROUP_VOLUME=/data1
export VLLM_USE_V1=1
export BASE_MODEL_PATH="/data1/minsoo3.kim/models/Qwen3-1.7B"
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
source scripts/setup_env.sh

bash cropi/scripts/run_cropi.sh full "$DATA_ROOT" "gsm8k/raw"
