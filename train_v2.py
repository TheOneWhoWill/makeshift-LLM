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
vocab_size = len(tokenizer)
print(f"Vocab size: {vocab_size}")
special_tokens_dict = {
    "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
tokenizer.add_special_tokens(special_tokens_dict)

vocab_size = len(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print(f"Vocab size: {vocab_size}")

context_size = 2048
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
	pad_token_id=tokenizer.pad_token_id,
)

model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Load dataset using datasets library
raw_dataset = load_dataset("allenai/c4", "en.noblocklist", split="train", streaming=True)

def tokenize_function(examples):
	return tokenizer(examples["text"])

tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names, # Remove old text columns
)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, though you could pad instead
    if total_length >= context_size:
        total_length = (total_length // context_size) * context_size
    
    # Split by chunks of context_size.
    result = {
        k: [t[i : i + context_size] for i in range(0, total_length, context_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


processed_dataset = tokenized_dataset.shuffle(buffer_size=1000).map(group_texts, batched=True)

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

trainer.train()
trainer.save_model(final_output_dir)