"""Builds the two Kaggle notebooks and their kernel-metadata.json files from the code in src/.

Usage: python make_notebooks.py <kaggle-username>
The notebooks embed src/*.py with %%writefile, so the code that runs on Kaggle is the code tested here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TRAIN_TITLE = "Finance Headline Sentiment DPO Training"
SERVE_TITLE = "Finance Headline Sentiment DPO Serving"
# Training versions passed the local CPU smoke test. vLLM's docs list T4 GPUs (compute capability 7.5) as supported.
TRAIN_PINS = "trl==1.16.0 peft==0.21.2 accelerate==1.15.0 transformers==5.20.2 datasets deepspeed wandb"
SERVE_PINS = "vllm datasets wandb"
NL = chr(10)
VERSIONS = '!pip list 2>/dev/null | grep -iE "^(torch|transformers|trl|peft|accelerate|deepspeed|vllm|wandb) "'


def slug(title):
    return title.lower().replace(" ", "-")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(name):
    return code(f"%%writefile {name}\n" + (SRC / name).read_text(encoding="utf-8"))


def notebook(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


STREAM = '''
import subprocess

def stream(cmd):
    """Runs a command and prints its output in this cell as it arrives. Returns the exit code."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    return p.wait()
'''

TRAIN_CELLS = [
    md("""
# Finance headline sentiment: DPO training

Fine-tunes Qwen2.5-0.5B-Instruct with DPO (Hugging Face TRL, LoRA) so it labels financial news headlines as
bearish, bullish or neutral, answering only in JSON. Training runs on both T4 GPUs with Accelerate and DeepSpeed
ZeRO-2 and is tracked in Weights & Biases. The serving notebook scores the result with vLLM on 500 held-out headlines.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(f"!pip install -q {TRAIN_PINS}\n# Kaggle ships torchao 0.10, and PEFT refuses to build LoRA layers while that old version is installed.\n!pip uninstall -y -q torchao\n{VERSIONS}"),
    code('''
import os
# W&B runs offline here and is synced afterwards, so no API key is stored in the notebook.
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "finsent-dpo"
os.environ["WANDB_DIR"] = "/kaggle/working"
'''),
    writefile("finsent.py"),
    code('''
import random
from datasets import load_dataset
from finsent import to_pair

random.seed(0)
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
train = ds["train"].shuffle(seed=42).select(range(2000))
pairs = train.map(to_pair, with_indices=True, remove_columns=train.column_names,
                  load_from_cache_file=False)  # the cache would keep pairs built with an older prompt
pairs.save_to_disk("/kaggle/working/dpo_train")
print(pairs[0])
print(pairs[1])
'''),
    writefile("train_dpo.py"),
    code('''
%%writefile ds_zero2.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: 4
  offload_optimizer_device: none
  offload_param_device: none
mixed_precision: fp16
num_machines: 1
num_processes: 2
machine_rank: 0
use_cpu: false
'''),
    code(STREAM + '''
import json

launchers = [
    ("DeepSpeed ZeRO-2 on 2 GPUs", ["accelerate", "launch", "--config_file", "ds_zero2.yaml", "train_dpo.py"]),
    ("Accelerate multi-GPU (DDP) on 2 GPUs",
     ["accelerate", "launch", "--multi_gpu", "--num_processes", "2", "--mixed_precision", "fp16", "train_dpo.py"]),
]
for label, cmd in launchers:
    print(f"=== {label}")
    if stream(cmd) == 0:
        json.dump({"launcher": label}, open("/kaggle/working/launcher.json", "w"))
        break
else:
    raise RuntimeError("Training failed with both launchers")
'''),
    code('''
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

merged = AutoPeftModelForCausalLM.from_pretrained("/kaggle/working/qwen-dpo", dtype=torch.float16).merge_and_unload()
merged.save_pretrained("/kaggle/working/qwen-dpo-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").save_pretrained("/kaggle/working/qwen-dpo-merged")
print(open("/kaggle/working/launcher.json").read())
print(open("/kaggle/working/qwen-dpo/run_info.json").read())
'''),
]

SERVE_CELLS = [
    md("""
# Finance headline sentiment: vLLM serving and evaluation

Serves the base model and the DPO model with vLLM on both T4 GPUs, then scores 500 held-out headlines with a
deterministic checker: valid-JSON rate, accuracy and throughput. A word-list challenger flags the answers a person
should review, the same design as Reconcile.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(
        "!nvidia-smi" + NL
        + "# uv picks the PyTorch build that matches this machine's GPU driver, as vLLM's install guide recommends." + NL
        + "!pip install -q uv" + NL
        + f"!timeout 900 uv pip install --system -q {SERVE_PINS} --torch-backend=auto" + NL
        + "# Kaggle's torchaudio is built for a different CUDA version, which stops vLLM from importing. vLLM doesn't use it." + NL
        + "!pip uninstall -y -q torchaudio" + NL
        + "!python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), 'GPUs')\"" + NL
        + VERSIONS),
    writefile("finsent.py"),
    writefile("run_vllm.py"),
    code('''
import glob, json, os, signal, subprocess

def run_limited(cmd, log_path, limit_seconds):
    """Runs cmd in its own process group with a hard time limit, writing its output to log_path."""
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=limit_seconds)
        except subprocess.TimeoutExpired:
            code = "timed out"
        try:
            os.killpg(p.pid, signal.SIGKILL)  # ends any vLLM worker processes still holding the GPUs
        except ProcessLookupError:
            pass
        p.wait()
    print(open(log_path, errors="replace").read()[-3000:])
    return code

config = glob.glob("/kaggle/input/**/qwen-dpo-merged/config.json", recursive=True)
assert config, "Add the training notebook's output as an input to this notebook."
models = {"base": "Qwen/Qwen2.5-0.5B-Instruct", "dpo": os.path.dirname(config[0])}

# Two T4s have no NVLink. Peer-to-peer over PCIe can stall NCCL on cloud VMs, so it is switched off.
os.environ["NCCL_P2P_DISABLE"] = "1"
for name, path in models.items():
    for tp in (2, 1):  # both GPUs first; one GPU if two-GPU serving fails or passes 10 minutes
        print(f"=== {name} with tensor_parallel_size={tp}")
        code = run_limited(["python", "run_vllm.py", name, path, str(tp)], f"/kaggle/working/vllm_{name}_tp{tp}.log", 600)
        print(f"--- exit: {code}")
        if code == 0:
            break
    else:
        raise RuntimeError(f"vLLM failed for {name}")
'''),
    code('''
import json
from finsent import summarize

results = {}
for name in ("base", "dpo"):
    d = json.load(open(f"/kaggle/working/outputs_{name}.json"))
    s = summarize(d["outputs"], d["gold"], d["texts"])
    s.update(headlines_per_second=d["headlines_per_second"], tensor_parallel_size=d["tensor_parallel_size"])
    results[name] = s
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)

print(f"{'':6}{'valid JSON':>12}{'accuracy':>10}{'review':>8}{'errors':>8}{'in review':>11}{'per sec':>9}")
for name, s in results.items():
    print(f"{name:6}{s['valid_json_rate']:>12.1%}{s['accuracy']:>10.1%}{s['review_rate']:>8.1%}"
          f"{s['errors']:>8}{s['errors_in_review']:>11}{s['headlines_per_second']:>9.1f}")
'''),
    code('''
import os, wandb

os.environ["WANDB_MODE"] = "offline"  # synced afterwards, so no API key is stored in the notebook
run = wandb.init(project="finsent-dpo", name="eval-500-headlines", dir="/kaggle/working")
cols = ["model", "valid_json_rate", "accuracy", "review_rate", "errors", "errors_in_review", "headlines_per_second"]
run.log({"results": wandb.Table(columns=cols, data=[[m] + [results[m][c] for c in cols[1:]] for m in results])})
run.finish()
'''),
]


def build(user):
    out = ROOT / "notebooks"
    for folder, title, cells, sources in (("train", TRAIN_TITLE, TRAIN_CELLS, []),
                                          ("serve", SERVE_TITLE, SERVE_CELLS, [f"{user}/{slug(TRAIN_TITLE)}"])):
        d = out / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{folder}.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        meta = {"id": f"{user}/{slug(title)}", "title": title, "code_file": f"{folder}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": sources, "model_sources": []}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print("wrote", d)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "KAGGLE_USERNAME")
