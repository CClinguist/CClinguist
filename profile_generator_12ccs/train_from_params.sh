#!/usr/bin/env bash

CONFIG_FILE="params_train.json"


train_model() {
  local cfg="$1"
  local acc=$(jq -r '.coef_acc_reward' <<< "$cfg")
  local obv=$(jq -r '.coef_obvious_reward' <<< "$cfg")
  local sch=$(jq -r '.coef_search_penalty' <<< "$cfg")
  local bp=$(jq -r '.base_penalty' <<< "$cfg")
  local dp=$(jq -r '.direction_penalty' <<< "$cfg")

  echo "Training with acc=$acc, obv=$obv, sch=$sch, bp=$bp, dp=$dp"
  CUDA_VISIBLE_DEVICES=1 python run.py \
    --coef_acc_reward $acc \
    --coef_obvious_reward $obv \
    --coef_search_penalty $sch \
    --base_penalty $bp \
    --direction_penalty $dp
}


len=$(jq length "$CONFIG_FILE")
for ((i=0; i<len; i++)); do
  cfg=$(jq ".[$i]" "$CONFIG_FILE")
  train_model "$cfg"
done
