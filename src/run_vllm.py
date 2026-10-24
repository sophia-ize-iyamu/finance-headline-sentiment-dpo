"""Serves one model with vLLM on the 500 held-out headlines and saves its raw answers and throughput.

Usage: python run_vllm.py <name> <model path or Hub id> <tensor_parallel_size>
Each model runs in its own process, so its GPU memory is fully released before the next one starts.
"""
import json
import os
import sys
import time

T0 = time.perf_counter()


def note(message):
    print(f"[{time.perf_counter() - T0:7.1f}s] {message}", flush=True)


note("importing vLLM")
from datasets import load_dataset  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from finsent import LABELS, PROMPT  # noqa: E402

name, path, tp = sys.argv[1], sys.argv[2], int(sys.argv[3])
val = load_dataset("zeroshot/twitter-financial-news-sentiment")["validation"].shuffle(seed=42).select(range(500))
convs = [[{"role": "user", "content": PROMPT + t}] for t in val["text"]]

note(f"starting vLLM for {name} on {tp} GPU(s)")
# enforce_eager skips torch.compile and CUDA graph capture. Two earlier runs stalled at start-up on the T4s.
llm = LLM(model=path, dtype="half", tensor_parallel_size=tp, seed=0, enforce_eager=True,
          disable_custom_all_reduce=True)
note("model loaded")
params = SamplingParams(temperature=0, max_tokens=32)
llm.chat(convs[:16], params)  # warm-up, so the timing below leaves out start-up costs
note("warm-up done")
start = time.perf_counter()
outs = llm.chat(convs, params)
seconds = time.perf_counter() - start

with open(f"/kaggle/working/outputs_{name}.json", "w") as fh:
    json.dump({"model": name, "path": path, "tensor_parallel_size": tp, "enforce_eager": True,
               "seconds": seconds, "headlines_per_second": len(convs) / seconds,
               "outputs": [o.outputs[0].text for o in outs],
               "texts": list(val["text"]), "gold": [LABELS[g] for g in val["label"]]}, fh)
note(f"{name}: {len(convs)} headlines in {seconds:.2f}s on {tp} GPU(s)")
# vLLM's worker processes keep Python alive after the run, so the script exits directly once results are saved.
sys.stdout.flush()
os._exit(0)
