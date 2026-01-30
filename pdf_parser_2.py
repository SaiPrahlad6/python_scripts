import pdfplumber
import pandas as pd
import pyspark

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
print(tables[0])
# Assuming a SparkSession named 'spark'
for tabl in tables:
  spark_df = spark.createDataFrame(tabl["df"])
  table_full_name = "main.default.my_new_table"   # Change to actual catalog information
  spark_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable(table_full_name)

