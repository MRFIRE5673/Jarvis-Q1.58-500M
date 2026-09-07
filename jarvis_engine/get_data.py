from datasets import load_dataset

# Small, permissively-licensed Python code subset
ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)

with open("data.txt", "w", encoding="utf-8") as f:
    count = 0
    for example in ds:
        code = example["content"]
        f.write(code + "\n\n")
        count += 1
        if count >= 2000:  # adjust based on how much data you want
            break

print(f"Wrote {count} code files to data.txt")