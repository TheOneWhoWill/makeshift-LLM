import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./rafikov_qwen2_finetuned_final"
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