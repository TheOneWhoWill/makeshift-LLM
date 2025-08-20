import os
from datasets import load_dataset

dataset = load_dataset("allenai/c4", "en.noblocklist", split="train", streaming=True)

target_size_gb = 385
target_size_bytes = target_size_gb * 1024**3
c4_sample_path = "./data/texts/c4/c4_sample.txt"

os.makedirs(os.path.dirname(c4_sample_path), exist_ok=True)

with open(c4_sample_path, "w", encoding="utf-8") as f:
	for i, row in enumerate(dataset):
		f.write(row["text"] + "\n")
		if i % 5000 == 0:
			size = os.path.getsize(c4_sample_path)
			print(f"Current size: {size / 1024**2:.2f} MB")
			if size >= target_size_bytes:
				break
