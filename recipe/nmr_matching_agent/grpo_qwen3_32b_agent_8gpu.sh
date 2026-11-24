#!/bin/bash
# M4 GRPO: multi-turn NMR matching tool agent (LoRA on ckpt-200 = M3 SFT, 8x A100 80GB).
#
# Adapted from recipe/nmr_matching/grpo_qwen3_32b_matching_8gpu.sh (single-turn version)
# by adding multi-turn tool-call rollout via tool_config_matching.yaml and swapping the
# reward function to reward_matching_agent.py (which parses <answer> from the transcript).
#
# IMPORTANT — before first run:
#   1) Build the agent-shape parquets (system+user JSON, not alpaca text):
#      cd recipe/nmr_matching_agent
#      python build_agent_parquet.py \
#          --from_parquet ../nmr_matching/data/rl_train_expanded.parquet \
#          --out data/rl_train_expanded_agent.parquet
#      python build_agent_parquet.py \
#          --from_alpaca /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json \
#          --out data/rl_val_1000_agent.parquet
#   2) Verify the prediction cache exists and has training candidates.
#      (recipe/nmr_matching_agent/prediction_cache.json — see M1 in README.)
#   3) Confirm ckpt-200 tokenizer files are present (see feedback-llamafactory-qwen3-tools).

set -x
cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT=$(pwd)

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /fs_mol/cxq/conda_env/OpenAgentRL

export VLLM_USE_V1=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= data / model / tools =================
train_data=${TRAIN_DATA:-$REPO_ROOT/recipe/nmr_matching_agent/data/rl_train_expanded_agent.parquet}
val_data=${VAL_DATA:-$REPO_ROOT/recipe/nmr_matching_agent/data/rl_val_1000_agent.parquet}
model_path=${MODEL_PATH:-/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-exp1/checkpoint-200}
tool_config_path=${TOOL_CONFIG_PATH:-$REPO_ROOT/recipe/nmr_matching_agent/tool_config_matching.yaml}
train_files="['$train_data']"
val_files="['$val_data']"

# ================= logging =================
project_name=${PROJECT_NAME:-nmr_matching_agent_grpo}
experiment_name=${EXPERIMENT_NAME:-qwen3-32b-matching-agent-grpo-lora}
default_local_dir=$REPO_ROOT/results/nmr_matching_agent/$experiment_name
mkdir -p "$default_local_dir"

# ================= algorithm =================
adv_estimator=grpo
reward_manager=naive
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=${KL_LOSS_COEF:-0.001}
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode=token-mean

# multi-turn tool rollout budget: ~6 turns should cover the M3 SFT distribution
# (median 3 turns per case, max 3 in test_1000).
max_turns=${MAX_TURNS:-6}
max_prompt_length=${MAX_PROMPT_LEN:-2048}
max_response_length=${MAX_RESP_LEN:-4096}   # multi-turn transcript longer than single-turn
actor_lr=${ACTOR_LR:-1e-6}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
n_resp_per_prompt=${N_RESP:-8}
n_resp_per_prompt_val=1

# ================= performance (8x A100 80GB) =================
infer_tp=${INFER_TP:-2}
train_sp=${TRAIN_SP:-2}
offload=True
use_lora=${USE_LORA:-1}
gpu_mem_util=${GPU_MEM_UTIL:-0.75}

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

lora_overrides=()
if [[ "$use_lora" == "1" ]]; then
  lora_overrides=(
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=32
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.model.use_shm=True
    actor_rollout_ref.rollout.layered_summon=True
    'actor_rollout_ref.actor.checkpoint.save_contents=[extra]'
  )
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$adv_estimator \
  algorithm.use_kl_in_reward=$use_kl_in_reward \
  algorithm.kl_ctrl.kl_coef=$kl_coef \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.return_raw_chat=True \
  data.train_batch_size=$train_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  custom_reward_function.path=recipe/nmr_matching_agent/reward_matching_agent.py \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=${reward_manager} \
  actor_rollout_ref.model.path=$model_path \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
  actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
  actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=${ROLLOUT_MODE:-async} \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem_util \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS:-128} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.top_k=50 \
  actor_rollout_ref.rollout.response_length=$max_response_length \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  ++actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
  actor_rollout_ref.ref.fsdp_config.param_offload=$offload \
  trainer.total_epochs=${TOTAL_EPOCHS:-3} \
  trainer.project_name=$project_name \
  trainer.experiment_name=$experiment_name \
  trainer.logger=['console'] \
  trainer.default_local_dir=$default_local_dir \
  trainer.default_hdfs_dir=null \
  trainer.resume_mode=${RESUME_MODE:-disable} \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-50} \
  trainer.test_freq=${TEST_FREQ:-40} \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPTS:-4} \
  trainer.critic_warmup=0 \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\" \
  "${lora_overrides[@]}"
