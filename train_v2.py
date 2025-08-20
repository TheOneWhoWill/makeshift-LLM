import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
	DataCollatorForLanguageModeling,
	Qwen2Config,
)
from datasets import load_dataset, IterableDataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.bos_token = "<|startoftext|>"
tokenizer.pad_token = "<|pad|>"
tokenizer.unk_token = "<|unk|>"

vocab_size = tokenizer.vocab_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print(f"Vocab size: {vocab_size}")

context_size = 1024 * 2
n_positions = context_size
n_layer = 12
n_head = 12
block_size = context_size
hidden_size = 768
checkpoint_dir = "gs://rafikov-qwen2-bucket/rafikov_qwen/"
final_output_dir = "gs://rafikov-qwen2-bucket/rafikov_qwen_final/"

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
	output_dir=checkpoint_dir,
	overwrite_output_dir=True,
	# I set the max_steps to 100000 but only trained to 80k-ish
	max_steps=100000,
	per_device_train_batch_size=8,
	gradient_accumulation_steps=8,
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

model = AutoModelForCausalLM.from_config(config)
model.to(device)

# Load dataset using datasets library
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
    create_token_blocks, gen_kwargs={"dataset_stream": dataset, "block_size": block_size}
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

trainer.train()
trainer.save_model(final_output_dir)