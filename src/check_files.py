from pathlib import Path
import os

data_dir = Path("data/raw")

print("Checking files in:", data_dir.absolute())
print("=" * 60)

for file in data_dir.glob("*.csv"):
    print(f"\nFile: {file.name}")
    print(f"  Size: {file.stat().st_size} bytes")
    
    try:
        with open(file, 'rb') as f:
            header = f.read(200)
            print(f"  First 200 bytes: {header}")
            print(f"  Is text: {b'\\x00' not in header}")  # Binary files have null bytes
    except Exception as e:
        print(f"  ERROR: {e}")