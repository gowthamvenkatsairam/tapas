python tapas/run_task_main.py \
  --task="SQA" \
  --output_dir="results" \
  --model_dir="ckpoints" \
  --train_batch_size=4 \
  --init_checkpoint="tapas_sqa_base/model.ckpt" \
  --bert_config_file="tapas_sqa_base/bert_config.json" \
  --mode="train"