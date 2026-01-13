{
  "chat_template": "chatml",
  "distributed_backend": "ddp",
  "mixed_precision": "fp16",
  "optimizer": "adamw_torch",
  "peft": "true",
  "scheduler": "linear",
  "unsloth": "true",
  "batch_size": "2",
  "block_size": "1024",
  "epochs": "1",
  "gradient_accumulation": "4",
  "lr": "0.0002",
  "model_max_length": "2048",
  "target_modules": "all-linear"
}