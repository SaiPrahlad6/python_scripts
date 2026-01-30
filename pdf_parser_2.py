import pdfplumber
import pandas as pd

def extract_tables_pdfplumber(pdf_path):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()

            for t_idx, table in enumerate(page_tables):
                df = pd.DataFrame(table)
                df.columns = df.iloc[0]   # first row as header
                df = df.iloc[1:].reset_index(drop=True)

                tables.append({
                    "pdf": pdf_path,
                    "page": page_num,
                    "table_index": t_idx,
                    "df": df
                })

    return tables


tables = extract_tables_pdfplumber("sample_table.pdf")
print(len(tables))
print(tables[0]["df"])
