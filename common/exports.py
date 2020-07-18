import pandas as pd
from common.file_system import ensure_directory


def save_to_excel(df, output_path, sheet_name="Sheet1", index=False):
    """Save the <df> as excel sheet"""
    ensure_directory(output_path)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        writer.save()


def save_to_excel_multi(dfs, output_path, sheet_names=None, index=False):
    ensure_directory(output_path)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for n, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=sheet_names[n] if sheet_names else f"Sheet{n}", index=index)
        writer.save()
