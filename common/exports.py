import pandas as pd
from common.file_system import ensure_directory


def save_to_excel(df, output_path, sheet_name="Sheet1", index=False):
    """Save the <df> as excel sheet"""
    ensure_directory(output_path)
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        writer.save()
