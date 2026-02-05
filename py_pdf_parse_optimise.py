import re
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
    first_row_text = sum(
        1 for v in first_row if isinstance(v, str) and v.strip() and not _is_number(v)
    )
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
        if len(non_empty) == 1:
            cell = row[non_empty[0]]
            if _looks_like_title_cell(cell):
                merged.append(row)
                continue
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


def _looks_like_title_cell(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", " ", value.replace("\u00A0", " ")).strip()
    if len(normalized) < 3:
        return False
    # Titles often contain words + a number/roman numeral suffix.
    if re.search(r"\b[A-Za-z][A-Za-z ]+\s+\d+\b", normalized):
        return True
    if re.search(r"\b[A-Za-z][A-Za-z ]+\s+[IVX]+\b", normalized):
        return True
    if not re.search(r"\d", normalized) and re.search(r"[A-Za-z]", normalized):
        return True
    return False


def _split_numeric_suffix_rows(rows: List[List[Any]]) -> List[List[Any]]:
    """
    Split cells that combine a numeric value with trailing title text.
    This prevents titles from being appended into the last numeric cell.
    """
    processed: List[List[Any]] = []
    for row in rows:
        split_done = False
        for col_idx, cell in enumerate(row):
            if isinstance(cell, str):
                match = re.match(r"^\s*([£$€]?\s*[\d.,()\-\u2212]+)\s+(.+)$", cell)
                if match:
                    numeric_part = match.group(1).strip()
                    trailing_text = match.group(2).strip()
                    if _is_number(numeric_part) and any(
                        ch.isalpha() for ch in trailing_text
                    ):
                        new_row = list(row)
                        new_row[col_idx] = numeric_part
                        extra = [None] * len(row)
                        extra[0] = trailing_text
                        processed.append(new_row)
                        processed.append(extra)
                        split_done = True
                        break
        if not split_done:
            processed.append(row)
    return processed


def _split_embedded_table_titles(rows: List[List[Any]]) -> List[List[Any]]:
    """
    If a cell contains "Game Theme + Table Title" (e.g., "DHD-00R Celestyal Journey"),
    split it into a separate title row + cleaned data row.
    """
    if not rows:
        return rows

    titles: List[str] = []
    for row in rows:
        non_empty = [v for v in row if v not in (None, "", " ")]
        if len(non_empty) == 1 and isinstance(non_empty[0], str):
            title = non_empty[0].strip()
            if len(title) > 2:
                titles.append(title)
        # Heuristic: a single value in col_1 after a series of nulls is a title.
        if (
            len(row) > 1
            and row[0] in (None, "", " ")
            and isinstance(row[1], str)
            and row[1].strip()
            and all(v in (None, "", " ") for v in row[2:])
        ):
            titles.append(row[1].strip())

    pattern_titles: List[str] = [
        r"[A-Za-z][A-Za-z ]+\s+\d+",
        r"[A-Za-z][A-Za-z ]+\s+[IVX]+",
    ]

    if not titles and not pattern_titles:
        return rows

    titles = sorted(set(titles), key=len, reverse=True)
    processed: List[List[Any]] = []

    for row in rows:
        split_done = False
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, str):
                continue
            # First try explicit titles discovered in the table.
            for title in titles:
                if not cell.endswith(title) or cell == title:
                    continue
                match = re.match(rf"^(.*\S)\s+{re.escape(title)}$", cell)
                if match:
                    prefix = match.group(1).strip()
                    title_row = [None] * len(row)
                    title_row[col_idx] = title
                    if not (
                        processed
                        and processed[-1][col_idx] == title
                        and all(
                            v in (None, "", " ") for i, v in enumerate(processed[-1]) if i != col_idx
                        )
                    ):
                        processed.append(title_row)
                    new_row = list(row)
                    new_row[col_idx] = prefix
                    processed.append(new_row)
                    split_done = True
                    break
            if split_done:
                break
            # Fallback to regex patterns when no explicit title was found.
            normalized = re.sub(r"\s+", " ", cell.replace("\u00A0", " ")).strip()
            for pattern in pattern_titles:
                match = re.search(rf"({pattern})\s*$", normalized)
                if match:
                    title = match.group(1).strip()
                    prefix = normalized[: match.start()].strip()
                    if not prefix:
                        continue
                    title_row = [None] * len(row)
                    title_row[col_idx] = title
                    if not (
                        processed
                        and processed[-1][col_idx] == title
                        and all(
                            v in (None, "", " ") for i, v in enumerate(processed[-1]) if i != col_idx
                        )
                    ):
                        processed.append(title_row)
                    new_row = list(row)
                    new_row[col_idx] = prefix
                    processed.append(new_row)
                    split_done = True
                    break
            if split_done:
                break
        if not split_done:
            processed.append(row)

    return processed


def _split_titles_after_null_run(rows: List[List[Any]], min_empty_run: int = 2) -> List[List[Any]]:
    """
    If col_1 (index 0) becomes non-empty after a run of empty values, and col_2
    contains a trailing title, split that trailing title into its own row.
    """
    if not rows:
        return rows
    processed: List[List[Any]] = []
    empty_run = 0
    title_pattern = re.compile(
        r"^(.*\S)\s+([A-Za-z][A-Za-z ]+\s+\d+|[A-Za-z][A-Za-z ]+\s+[IVX]+)\s*$"
    )

    for row in rows:
        col0 = row[0] if len(row) > 0 else None
        col1 = row[1] if len(row) > 1 else None
        col0_empty = col0 in (None, "", " ")

        if col0_empty:
            empty_run += 1
            processed.append(row)
            continue

        # col_1 has a value after a run of nulls: check col_2 for an appended title.
        if empty_run >= min_empty_run and isinstance(col1, str):
            normalized = re.sub(r"\s+", " ", col1.replace("\u00A0", " ")).strip()
            match = title_pattern.match(normalized)
            if match:
                prefix = match.group(1).strip()
                title = match.group(2).strip()
                new_row = list(row)
                new_row[1] = prefix
                processed.append(new_row)

                title_row = [None] * len(row)
                title_row[1] = title
                processed.append(title_row)
            else:
                processed.append(row)
        else:
            processed.append(row)

        empty_run = 0

    return processed


def _drop_duplicate_header_columns(rows: List[List[Any]]) -> List[List[Any]]:
    if not rows:
        return rows
    header_row = None
    for row in rows:
        lowered = [
            str(v).strip().lower()
            for v in row
            if isinstance(v, str) and v.strip()
        ]
        if "game theme" in lowered and "coin-in" in lowered and "coin-out" in lowered:
            header_row = row
            break

    if header_row is None:
        return rows

    seen: Dict[str, int] = {}
    drop_idx: List[int] = []
    for idx, val in enumerate(header_row):
        if not isinstance(val, str):
            continue
        label = val.strip().lower()
        if not label:
            continue
        if label in seen:
            drop_idx.append(idx)
        else:
            seen[label] = idx

    if not drop_idx:
        return rows

    deduped: List[List[Any]] = []
    for row in rows:
        deduped.append([v for i, v in enumerate(row) if i not in drop_idx])
    return deduped


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
            cleaned_rows = _split_titles_after_null_run(cleaned_rows)
            cleaned_rows = _split_embedded_table_titles(cleaned_rows)
            cleaned_rows = _split_numeric_suffix_rows(cleaned_rows)
            cleaned_rows = _drop_duplicate_header_columns(cleaned_rows)

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
            df.insert(0, "page", page_index + 1)
            df.insert(1, "row_number", range(1, len(df) + 1))

            results.append(
                {
                    "page": page_index + 1,
                    "table": table_index,
                    "dataframe": df,
                }
            )

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

                pending = {
                    "page": page_index + 1,
                    "table": table_index,
                    "columns": df.columns.tolist(),
                    "bbox": table_bbox,
                    "page_height": page.rect.height,
                    "dataframe": df,
                }

    return results


def _export_tables(results: Iterable[Dict[str, Any]], prefix: str, output_dir: str) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    for item in results:
        page = item["page"]
        table = item["table"]
        df = item["dataframe"]
        safe_prefix = prefix.replace(":", "_").replace("/", "_").replace("\\", "_")
        output_path = os.path.join(
            output_dir, f"{safe_prefix}_page_{page}_table_{table}.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"\n{prefix} | Page {page} | Table {table}")
        print(df.head(5))
        print("-" * 100)


def _export_combined(results: Iterable[Dict[str, Any]], prefix: str, output_dir: str) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    max_data_cols = 0
    for item in results:
        df = item["dataframe"]
        data_col_indices = [
            i
            for i, c in enumerate(df.columns)
            if c not in ("page", "row_number")
        ]
        max_data_cols = max(max_data_cols, len(data_col_indices))

    for item in results:
        table = item["table"]
        df = item["dataframe"].copy()
        data_col_indices = [
            i
            for i, c in enumerate(df.columns)
            if c not in ("page", "row_number")
        ]
        df_data = df.iloc[:, data_col_indices]
        # Align by column position, not by header names.
        df_data.columns = [f"col_{i+1}" for i in range(len(df_data.columns))]
        df_data = df_data.reindex(
            columns=[f"col_{i+1}" for i in range(max_data_cols)]
        )
        combined_df = pd.concat([df[["page", "row_number"]], df_data], axis=1)
        combined_df.insert(0, "table", table)
        frames.append(combined_df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    safe_prefix = prefix.replace(":", "_").replace("/", "_").replace("\\", "_")
    output_path = os.path.join(output_dir, f"{safe_prefix}_combined.csv")
    combined.to_csv(output_path, index=False)


def _make_columns_unique(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for col in columns:
        base = col if col else "col"
        count = seen.get(base, 0)
        if count == 0:
            unique.append(base)
        else:
            unique.append(f"{base}_{count}")
        seen[base] = count + 1
    return unique


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
    from datetime import datetime
    output_dir = f"outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parsed = pdf_table_parser_to_dataframe(
        pdf,
        table_settings=TABLE_SETTINGS,
        header="auto",
        merge_wrapped_rows=True,
        merge_across_pages=True,
    )
    _export_tables(parsed, prefix=pdf.replace(".pdf", ""), output_dir=output_dir)
    _export_combined(parsed, prefix=pdf.replace(".pdf", ""), output_dir=output_dir)