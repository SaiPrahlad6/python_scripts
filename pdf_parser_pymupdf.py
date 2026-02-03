import pymupdf
import pandas as pd

doc = pymupdf.open("complex_exam.pdf")

for i in range(len(doc)):
    page = doc[i]
    tables = page.find_tables()
    for idx, table in enumerate(tables, 1):
        df = table.to_pandas()
        
        # Clean all cells - iterate through each cell manually
        for col_idx, col in enumerate(df.columns):
            for row_idx in range(len(df)):
                val = df.iat[row_idx, col_idx]
                if isinstance(val, str):
                    cleaned = val.replace('\n_ _', '').replace('\n_', '').replace('\n', ' ').strip()
                    df.iat[row_idx, col_idx] = cleaned
        
        # Clean column names
        df.columns = [col.replace('\n_ _', '').replace('\n_', '').replace('\n', ' ').strip() for col in df.columns]
        
        print(f"\nTable {idx}:")
        print(df.to_string(index=False))
        print("-" * 100)