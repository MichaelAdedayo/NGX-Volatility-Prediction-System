from pathlib import Path
import pandas as pd

data_dir = Path(r"C:\Users\DREAMUSER\Documents\NSE Volatility Project\NSE_Volatility_Project\data\raw")

for csv_file in sorted(data_dir.glob("*.csv")):
    print(f"\n{'='*50}")
    print(f"File: {csv_file.name}")
    print(f"Size: {csv_file.stat().st_size} bytes")
    
    try:
        # Try reading first few lines as text
        with open(csv_file, 'rb') as f:
            raw = f.read(200)
        print(f"First 200 bytes (raw): {raw[:100]}")
        
        # Try pandas
        df = pd.read_csv(csv_file, nrows=3)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"First row:\\n{df.iloc[0]}")
        print("✅ READABLE")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")