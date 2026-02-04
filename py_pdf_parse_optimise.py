import pymupdf
import pandas as pd
from typing import Iterable, List, Dict, Any, Optional


def _clean_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return (
        value.replace("\n_ _", "")
        .replace("\n_", "")
        .replace("\n", " ")
        .strip()
    )


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.replace(",", "").replace("(", "").replace(")", "").strip()
        if text == "":
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def _looks_like_header(first_row: List[Any], body: List[List[Any]]) -> bool:
    if not first_row:
        return False
    first_row_text = sum(1 for v in first_row if isinstance(v, str) and v.strip())
    if first_row_text == 0:
        return False
    # If most of the first row is text and the next rows have numbers, treat as header.
    if not body:
        return True
    numeric_in_body = sum(
        1 for row in body[:3] for v in row if _is_number(v)
    )
    return first_row_text >= max(1, len(first_row) // 2) and numeric_in_body > 0


def _merge_wrapped_rows(rows: List[List[Any]]) -> List[List[Any]]:
    if not rows:
        return rows
    merged: List[List[Any]] = []
    for row in rows:
        non_empty = [i for i, v in enumerate(row) if v not in (None, "", " ")]
        if (
            merged
            and len(non_empty) <= 2
            and all(i < len(merged[-1]) for i in non_empty)
        ):
            # If a row looks like a continuation, merge into previous row.
            for i in non_empty:
                prev = merged[-1][i]
                cur = row[i]
                if prev in (None, "", " "):
                    merged[-1][i] = cur
                else:
                    merged[-1][i] = f"{prev} {cur}".strip()
            continue
        merged.append(row)
    return merged


def _normalize_header(header: List[Any]) -> List[str]:
    return [str(v).strip().lower() if v not in (None, "", " ") else "" for v in header]


def _row_matches_header(row: List[Any], header: List[Any]) -> bool:
    if not row or not header or len(row) != len(header):
        return False
    return _normalize_header(row) == _normalize_header(header)


def _table_near_top(table_bbox: Any, page_height: float, threshold: float = 0.12) -> bool:
    # table_bbox = (x0, y0, x1, y1)
    return table_bbox and (table_bbox[1] / page_height) <= threshold


def _table_near_bottom(table_bbox: Any, page_height: float, threshold: float = 0.12) -> bool:
    return table_bbox and ((page_height - table_bbox[3]) / page_height) <= threshold


def pdf_table_parser_to_dataframe(
    pdf_path: str,
    *,
    table_settings: Optional[Dict[str, Any]] = None,
    header: str = "auto",
    merge_wrapped_rows: bool = True,
    merge_across_pages: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parse tables from a PDF into DataFrames, preserving column structure.
    Returns a list of dicts with page/table metadata + dataframe.
    """
    doc = pymupdf.open(pdf_path)
    results: List[Dict[str, Any]] = []
    settings = table_settings or {}
    pending: Optional[Dict[str, Any]] = None

    for page_index in range(len(doc)):
        page = doc[page_index]
        tables = page.find_tables(**settings)

        for table_index, table in enumerate(tables, 1):
            raw_rows = table.extract()  # list of rows
            cleaned_rows = [[_clean_cell(v) for v in row] for row in raw_rows]

            if merge_wrapped_rows:
                cleaned_rows = _merge_wrapped_rows(cleaned_rows)

            max_cols = max((len(r) for r in cleaned_rows), default=0)
            padded_rows = [r + [None] * (max_cols - len(r)) for r in cleaned_rows]

            if header in ("first", "auto") and padded_rows:
                first_row = padded_rows[0]
                body = padded_rows[1:]
                if header == "first" or _looks_like_header(first_row, body):
                    columns = [
                        c if c not in (None, "", " ") else f"col_{i+1}"
                        for i, c in enumerate(first_row)
                    ]
                    data = body
                else:
                    columns = [f"col_{i+1}" for i in range(max_cols)]
                    data = padded_rows
            else:
                columns = [f"col_{i+1}" for i in range(max_cols)]
                data = padded_rows

            df = pd.DataFrame(data, columns=columns)
            df = df.dropna(axis=1, how="all")

            if merge_across_pages:
                table_bbox = getattr(table, "bbox", None)
                is_top = _table_near_top(table_bbox, page.rect.height)
                is_bottom = (
                    pending is not None
                    and _table_near_bottom(pending["bbox"], pending["page_height"])
                )

                if (
                    pending is not None
                    and pending["page"] == page_index
                    and is_top
                    and is_bottom
                    and _normalize_header(df.columns.tolist())
                    == _normalize_header(pending["columns"])
                ):
                    # Drop repeated header row if it matches the existing header.
                    if _row_matches_header(df.iloc[0].tolist(), pending["columns"]):
                        df = df.iloc[1:].reset_index(drop=True)
                    pending["dataframe"] = pd.concat(
                        [pending["dataframe"], df], ignore_index=True
                    )
                    continue

                if pending is not None:
                    results.append(pending["result"])
                    pending = None

                pending = {
                    "page": page_index + 1,
                    "table": table_index,
                    "columns": df.columns.tolist(),
                    "bbox": table_bbox,
                    "page_height": page.rect.height,
                    "dataframe": df,
                    "result": {
                        "page": page_index + 1,
                        "table": table_index,
                        "dataframe": df,
                    },
                }
            else:
                results.append(
                    {
                        "page": page_index + 1,
                        "table": table_index,
                        "dataframe": df,
                    }
                )

    if pending is not None:
        results.append(pending["result"])

    return results


def _export_tables(results: Iterable[Dict[str, Any]], prefix: str) -> None:
    for item in results:
        page = item["page"]
        table = item["table"]
        df = item["dataframe"]
        df.to_csv(f"{prefix}_page_{page}_table_{table}.csv", index=False)
        print(f"\n{prefix} | Page {page} | Table {table}")
        print(df.head(5))
        print("-" * 100)


TABLE_SETTINGS = {
    # Lines-based detection helps match PDF layout for these reports.
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
}

pdfs = [
    "IGT December_25 report.pdf",
    "IGT Silversea - Tui December_25 report.pdf",
]

for pdf in pdfs:
    parsed = pdf_table_parser_to_dataframe(
        pdf,
        table_settings=TABLE_SETTINGS,
        header="auto",
        merge_wrapped_rows=True,
        merge_across_pages=True,
    )
    _export_tables(parsed, prefix=pdf.replace(".pdf", ""))