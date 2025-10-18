# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#export LAYERNORM_TYPE=fast_layernorm
#export USE_DEEPSPEED_EVO_ATTENTION=true
# wget -P /data/af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt

#==================== for original model ====================
#export CUDA_VISIBLE_DEVICES=3
#checkpoint_path="/data/af3-dev/release_model/model_v0.2.0.pt"
#
#PYTHONPATH=. python3 ./runner/train.py \
#--run_name protenix_finetune \
#--eval_first True \
#--seed 42 \
#--base_dir /workspace/wuhaixu/output_alphafold \
#--dtype bf16 \
#--project protenix \
#--use_wandb false \
#--diffusion_batch_size 48 \
#--eval_interval 1 \
#--log_interval 50 \
#--checkpoint_interval 1 \
#--ema_decay 0.999 \
#--train_crop_size 384 \
#--max_steps 1 \
#--warmup_steps 2000 \
#--model.N_cycle 3 \
#--lr 0.001 \
#--sample_diffusion.N_step 20 \
#--load_checkpoint_path ${checkpoint_path} \
#--load_ema_checkpoint_path ${checkpoint_path} \
#--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
#--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list examples/finetune_subset.txt \
#--data.test_sets recentPDB_1536_sample384_0925

#==================== for flashbias model ====================
export CUDA_VISIBLE_DEVICES=2
checkpoint_path="path to the finetuned model"

PYTHONPATH=. python3 ./runner/train.py \
--run_name protenix_finetune \
--eval_first True \
--seed 42 \
--base_dir /workspace/wuhaixu/output_alphafold_flashbias_16steps \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 48 \s
--eval_interval 1 \
--log_interval 50 \
--checkpoint_interval 1 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 1 \
--warmup_steps 2000 \
--model.N_cycle 3 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list examples/finetune_subset.txt \
--data.test_sets recentPDB_1536_sample384_0925

