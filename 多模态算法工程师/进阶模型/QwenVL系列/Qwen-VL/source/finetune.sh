MODEL_PATH="pretrained_ckpt/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
DATA_PATH="/home/user/Documents/flj/My_Projects/Multi-model-Carrer-Finding/datasets/train_data.json"
PYTHON="/home/user/anaconda3/envs/qwenvl/bin/python"

CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 $PYTHON finetune.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir ./output_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --learning_rate 2e-4 \
  --fp16 True \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --model_max_length 2048