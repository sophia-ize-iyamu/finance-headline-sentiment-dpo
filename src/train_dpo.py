"""DPO fine-tuning of Qwen2.5-0.5B-Instruct with LoRA, launched by Accelerate (DeepSpeed ZeRO-2 on Kaggle).

Set SMOKE=1 for a two-step CPU run that only checks the code path.
"""
import json
import os

from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/working/dpo_train")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/qwen-dpo")
SMOKE = os.environ.get("SMOKE") == "1"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)  # fp32 master weights; fp16 mixed precision on the T4s
train = load_from_disk(DATA_DIR)
if SMOKE:
    train = train.select(range(2))

args = DPOConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1 if SMOKE else 4,
    gradient_accumulation_steps=1 if SMOKE else 4,
    learning_rate=5e-5,
    num_train_epochs=1,
    beta=0.1,
    # Run 1 used DPO alone, and the chosen answers' likelihood fell along with the rejected ones. The added SFT
    # term trains on the chosen answers directly, so the right label in valid JSON is rewarded as well as preferred.
    loss_type=["sigmoid", "sft"],
    loss_weights=[1.0, 1.0],
    max_length=256,
    bf16=False,  # TRL turns bf16 on by default, but T4 GPUs don't support it
    fp16=not SMOKE,
    use_cpu=SMOKE,
    logging_steps=5,
    save_strategy="no",
    report_to="none" if SMOKE else "wandb",
    run_name="qwen0.5b-dpo-sft-finsent",
    seed=42,
    max_steps=2 if SMOKE else -1,
)
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
# With a LoRA adapter, TRL uses the frozen base model as the reference, so no second copy is loaded.
trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=train,
                     processing_class=tok, peft_config=peft_config)
trainer.train()
trainer.save_model(OUT_DIR)
if trainer.accelerator.is_main_process:
    with open(os.path.join(OUT_DIR, "log_history.json"), "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=1)
    with open(os.path.join(OUT_DIR, "run_info.json"), "w") as fh:
        json.dump({"num_processes": trainer.accelerator.num_processes,
                   "distributed_type": str(trainer.accelerator.distributed_type),
                   "train_rows": len(train), "global_step": trainer.state.global_step}, fh, indent=1)
