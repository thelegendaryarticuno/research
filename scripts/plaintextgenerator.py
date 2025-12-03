import os

def generate_plaintexts(sizes_kb, num_samples=5, out_dir="dataset/plaintexts"):
    os.makedirs(out_dir, exist_ok=True)
    for size in sizes_kb:
        for i in range(num_samples):
            data = os.urandom(size * 1024)  # random bytes
            fname = f"{size}KB_{i}.bin"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(data)

generate_plaintexts([16, 64, 256, 512, 1024, 2048], num_samples=10)
