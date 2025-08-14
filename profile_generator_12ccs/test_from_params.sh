#!/usr/bin/env bash
# cd to the directory where this script is located and run it

CONFIG_FILE="params_base.json"

# CONFIG_FILE="params_test.json"


run_experiments() {
  local cfg="$1"

  local acc=$(jq -r '.coef_acc_reward' <<< "$cfg")
  local obv=$(jq -r '.coef_obvious_reward' <<< "$cfg")
  local sch=$(jq -r '.coef_search_penalty' <<< "$cfg")
  local bp=$(jq -r '.base_penalty' <<< "$cfg")
  local dp=$(jq -r '.direction_penalty' <<< "$cfg")
  local load_path=$(jq -r '.load_path' <<< "$cfg")


  for idx in {0..8}; do
    echo "Running with acc=$acc, obv=$obv, sch=$sch, bp=$bp, dp=$dp, idx=$idx"
    CUDA_VISIBLE_DEVICES=1 python run.py \
      --coef_acc_reward $acc \
      --coef_obvious_reward $obv \
      --coef_search_penalty $sch \
      --base_penalty $bp \

      --direction_penalty $dp \
      --indices_to_add $idx\
      --load_path $load_path  
  done
}


len=$(jq length $CONFIG_FILE)
for ((i=0; i<len; i++)); do
  
  cfg=$(jq ".[$i]" $CONFIG_FILE)
  run_experiments "$cfg"
done

python result_analyze.py    #see the results in the results_12CCs_coef/analyze_final directory
python tree_visualize.py    #see the tree visualization in the results_12CCs_coef/tree_visualization directory

