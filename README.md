---
license: apache-2.0
datasets:
- allenai/c4
- databricks/databricks-dolly-15k
language:
- en
pipeline_tag: text-generation
tags:
- qwen2
- transformers
- text-generation
---
# Makeshift Qwen2

## Introduction
Recently I've been interested in LLMs and wanted to train my own from scratch using the Qwen2 architecture provided through the Hugging Face transformers library. This was created locally on my personal laptop and is not powerful enough to be useful in any way, but it can respond to simple queries. I would recommend using a better-trained lightweight model instead of this one, as I've observed that although explicit in your queries, it often hallucinates data such as fictional U.S. Presidents or starts ranting about Chicago when told "Hey". The only advantage I can point out is its small size, weighing in at only 203 MB. This is just the Github for the code to create the model, you can use it on [Huggingface](https://huggingface.co/TheOneWhoWill/makeshift-qwen2) through the ```run.py``` script or with your own.

## Model Details
- **Model Name:** Makeshift LLM
- **Architecture:** Qwen2-based
- **Context:** 1024 Tokens
- **Vocab Size:** 32,000 tokens
- **Qwen2 Specific:** Hidden size of 768, 6 layers, 6 heads

## Training Details
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU
- **Cuda:** CUDA was used during pre-training and fine-tuning.
- **VRAM:** 8gb

A 28.4 GB subset of the [AllenAI C4 English](https://huggingface.co/datasets/allenai/c4) dataset was used for pre-training as well as for generating the tokenizer. However, the model was only trained up to an epoch of 0.77 (77% complete) because the loss was very stable at 3.5, and I didn't see any reason to continue training. Pre-training took about 18.5 hours with the GPU overclocked to its maximum capacity. Post-training involved 6 epochs of [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) formatted in ChatML with 50 random possible system prompts.

## How to use
Here below I created a simple python script you can use. The model should be usable directly through the transformers library but you can change the model path to point to a directory containing the model too.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "TheOneWhoWill/makeshift-qwen2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
	model_path,
	torch_dtype="auto",
	device_map="auto"
)

from transformers import pipeline

pipe = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer
)

messages = [
	{"role": "system", "content": "You are a helpful AI assistant. Always provide clear, accurate, and concise answers."}
]

while True:
	user_input = input("User: ")
	if user_input.lower() in ["exit", "quit"]:
		print("Exiting the chat.")
		break
	messages.append({"role": "user", "content": user_input})
	# Generate and print
	response = pipe(
		messages,
		max_new_tokens=256,
		do_sample=True,
		temperature=0.7,
		top_k=50,
		top_p=0.95
	)
	response = response[0]['generated_text'][-1]["content"]
	messages.append({"role": "assistant", "content": response})
	print("Assistant:", response)
```