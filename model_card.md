---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
- zeroshot/twitter-financial-news-sentiment
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- dpo
- trl
- lora
- finance
- sentiment-analysis
---

# Qwen2.5-0.5B finance headline sentiment (DPO plus SFT)

Qwen2.5-0.5B-Instruct, fine-tuned to label financial news headlines as bearish, bullish or neutral, with every answer in JSON such as `{"label": "neutral"}`. I trained it with TRL's DPO loss plus an SFT term, using a LoRA adapter that I then merged into the base weights.

## Results on 500 held-out headlines

| | Base model | This model |
|---|---|---|
| Valid JSON | 99.6% | 100.0% |
| Accuracy | 38.2% | 84.6% |
| Accuracy on bearish headlines (78) | 56.4% | 61.5% |
| Accuracy on bullish headlines (96) | 88.5% | 64.6% |
| Accuracy on neutral headlines (326) | 19.0% | 96.0% |

Answering neutral every time would be right 65.2% of the time on this set. Both models were served with vLLM 0.32.0 on 2 T4 GPUs with greedy decoding. An answer was counted only if all of it parsed as JSON with an allowed label.

## Usage

Put the headline after the training prompt.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "sophia-ize-iyamu/qwen2.5-0.5b-finsent-dpo"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

prompt = ('Classify the sentiment of this financial news headline as bearish, bullish or neutral. '
          'Reply with JSON only, for example {"label": "neutral"}.\n\n')
messages = [{"role": "user", "content": prompt + "Acme beats estimates and raises full-year guidance"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))  # {"label": "bullish"}
```

## Training

- **Data:** 2,000 preference pairs from the training split of zeroshot/twitter-financial-news-sentiment (MIT). Each chosen answer is the correct label in JSON. The rejected answers are a wrong label in JSON for 1,000 pairs, and a sentence with the correct label for the other 1,000.
- **Method:** TRL 1.16.0 `DPOTrainer` with `loss_type=["sigmoid", "sft"]`, weighted equally. LoRA rank 16 on all linear layers, beta 0.1, learning rate 5e-5, 1 epoch, 32 pairs per step, 63 steps.
- **Hardware:** 2 T4 GPUs on Kaggle with Accelerate and DeepSpeed ZeRO-2 in fp16. Training took 133.5 seconds.
- **Earlier run:** with DPO alone and the labels written in the prompt as `{"label": "bearish" | "bullish" | "neutral"}`, valid JSON fell from 35.2% to 4.2%. Of the base model's 500 answers, 306 were `{"label": "bearish}`, with a quote missing, so I wrote a clearer prompt. I also added the SFT term, because the chosen answers' implicit reward had fallen to -0.86 by step 60.

## Limitations

- I trained and tested it on short finance tweets only, so longer news text is untested.
- Accuracy is uneven across labels: 96.0% on neutral headlines, 61.5% on bearish and 64.6% on bullish.
- I wrote this prompt after reading the earlier run's answers to the same 500 test headlines, and I checked it on test headlines 1 to 40.
- The rejected answers are synthetic and simpler than a deployed model's errors.
- It's for research and learning only. Trading decisions are out of scope.
