import torch
import os
from tokenizers import Tokenizer
from transformers import (
	AutoModelForCausalLM,
	PreTrainedTokenizerFast,
	Trainer,
	TrainingArguments,
	DataCollatorForLanguageModeling,
	Qwen2Config,
)
from datasets import load_dataset, IterableDataset
from transformers.trainer_utils import get_last_checkpoint

# Load tokenizer from tokenizer.json
custom_tokenizer = Tokenizer.from_file("tokenizer.json")

tokenizer = PreTrainedTokenizerFast(tokenizer_object=custom_tokenizer)
tokenizer.eos_token = "<|endoftext|>"
tokenizer.bos_token = "<|startoftext|>"
tokenizer.pad_token = "<|pad|>"
tokenizer.unk_token = "<|unk|>"

vocab_size = tokenizer.vocab_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print(f"Vocab size: {vocab_size}")

n_positions = 1024
n_layer = 6
n_head = 6
block_size = 1024
hidden_size = 768

config = Qwen2Config(
	vocab_size=vocab_size,
	max_position_embeddings=n_positions,
	num_hidden_layers=n_layer,
	num_attention_heads=n_head,
	num_key_value_heads=n_head,
	hidden_size=hidden_size,
	intermediate_size=hidden_size * 4,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id,
)

training_args = TrainingArguments(
	output_dir="./rafikov_qwen2_v2_output",
	overwrite_output_dir=True,
	# I set the max_steps to 100000 but only trained to 80k-ish
	max_steps=100000,
	per_device_train_batch_size=8,
	gradient_accumulation_steps=2,
	fp16=True,
	learning_rate=5e-5,
	max_grad_norm=1.0,
	weight_decay=0.01,
	lr_scheduler_type="cosine",
	warmup_ratio=0.01,
	save_strategy="steps",
	save_steps=100,
	# Only save 3 previous iterations
	save_total_limit=3,
	logging_steps=10,
	ignore_data_skip=True,
)

# Default to 0 samples to skip
num_samples_to_skip = 0
last_checkpoint = get_last_checkpoint(training_args.output_dir)

if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    try:
        import json
        state_path = os.path.join(last_checkpoint, "trainer_state.json")
        with open(state_path, "r") as f:
            state = json.load(f)
        
        # This calculation is correct for single-GPU training.
        num_samples_to_skip = state["global_step"] * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        print(f"Will skip {num_samples_to_skip} samples from the dataset.")

    except (FileNotFoundError, KeyError):
        print("Could not load trainer state to determine how many samples to skip. Starting dataset from the beginning.")
        last_checkpoint = None # Don't resume if we can't determine skip amount.

model = AutoModelForCausalLM.from_config(config)
model.to(device)

# Load dataset using datasets library
# dataset = load_dataset("text", data_files={"train": "./data/merged/output.md"}, streaming=True)["train"]
dataset = load_dataset("text", data_files={"train": "./data/texts/c4/c4_sample.txt"}, streaming=True)["train"]

def tokenize_function(examples):
	return tokenizer(examples["text"])

def create_token_blocks(dataset_stream, block_size, num_to_skip=0):
	# Buffer for concatenated token IDs
	buffer = []
	
	# Counter for blocks yielded
	blocks_yielded = 0
	last_log_step = 0

	print("Starting dataset processing...")
	for item in dataset_stream:
		# Tokenize each text individually
		tokenized_text = tokenize_function(item)
		buffer.extend(tokenized_text['input_ids'])
		
		# Yield blocks of the specified size
		while len(buffer) >= block_size:
			if blocks_yielded < num_to_skip:
				# We are in the skipping phase
				buffer = buffer[block_size:]
				blocks_yielded += 1
				if blocks_yielded % 1000 == 0 and blocks_yielded > last_log_step:
					print(f"Skipped {blocks_yielded}/{num_to_skip} blocks...")
					last_log_step = blocks_yielded
				continue

			if blocks_yielded == num_to_skip and num_to_skip > 0:
				print(f"Finished skipping {num_to_skip} blocks. Starting training.")
				# To avoid printing this message on every subsequent block
				num_to_skip = -1 

			block = buffer[:block_size]
			buffer = buffer[block_size:]
			yield {"input_ids": block, "labels": block.copy()}

processed_dataset = IterableDataset.from_generator(
    create_token_blocks, gen_kwargs={"dataset_stream": dataset, "block_size": block_size, "num_to_skip": num_samples_to_skip}
)

data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False,
)

trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=processed_dataset,
)

trainer.tokenizer.model_max_length = block_size
trainer.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# When using an IterableDataset that we manually fast-forward, we pass
# `resume_from_checkpoint` to load the model/optimizer/scheduler state
# and `ignore_data_skip=True` to prevent the Trainer from skipping batches
# in the dataloader (since we do it ourselves).
if last_checkpoint:
	trainer.train(resume_from_checkpoint=last_checkpoint)
else:
	trainer.train()
trainer.save_model("./rafikov_qwen2_v2")