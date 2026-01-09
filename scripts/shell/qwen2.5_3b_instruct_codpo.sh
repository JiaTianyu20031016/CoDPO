CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file trl/accelerate_configs/zero3.yaml scripts/run_codpo.py \
    --dataset_name datasets/ultrafeedback_binarized_annotated \
    --model_name_or_path /data1/Common_LLM_Base/Qwen/Qwen2.5-3B-Instruct/ \
    --learning_rate 5.0e-7 \
    --beta 0.1 \
    --head_warmup_steps 0 \
    --value_loss_coef 0.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_strategy epoch \
    --output_dir checkpoints/Qwen2.5-3B-Instruct-CODPO-no-warmup \
    --no_remove_unused_columns \
    # --precompute_ref_log_probs