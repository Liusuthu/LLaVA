python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/llava-v1.5-13b-task-lora \
    --model-base lmsys/vicuna-13b-v1.5 \
    --save-model-path ./llava-merge-model