python tapas/tapas/run_task_main.py \
  --task="SQA" \
  --output_dir="results" \
  --model_dir="ckpoints4" \
  --init_checkpoint="ckpoints4/model.ckpt-1312" \
  --bert_config_file="tapas_sqa_base/bert_config.json" \
  --mode="predict_and_evaluate"