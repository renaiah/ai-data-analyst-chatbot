import pandas as pd
import numpy as np
import os
import time
import argparse

# NOTE : use this command to upload Dataset : python3 uploadDataset.py --csv data/test1.csv --out store   --> test1 is a dataset name
# Data Preprocessing
def preprocess_dataframe(df):
    numeric_cols = [
        "ValuePlus", "IsOffline", "SaleValue_Actual",
        "SaleValue_Actual_E_Tax", "SaleValue_CP_E_Tax",
        "SaleValue_CP_E_Tax_Zero_Replacement", "SaleValue_MRP", "Quantity"
    ]
    # text_cols = ["StoreID", "State", "SKU", "Day", "Systime"]
    text_cols = ["StoreID", "Day", "Systime"]

    # Replace empty strings with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Clean numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Clean text columns
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").replace("", "unknown")

    print("Cleaned missing values (numeric → 0, text → 'unknown')")
    return df


def saveDataset(csv_path, out_dir="store"):
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading new CSV data from: {csv_path}")
    new_df = pd.read_csv(csv_path)
    print(f"New rows loaded: {len(new_df)}")

    new_df = preprocess_dataframe(new_df)

    merged_csv = os.path.join(out_dir, "merged_data.csv")

    # Merge with existing data if file exists
    if os.path.exists(merged_csv):
        old_df = pd.read_csv(merged_csv)
        merged_df = pd.concat([old_df, new_df], ignore_index=True)
        print(f"Total merged rows: {len(merged_df)}")
    else:
        merged_df = new_df.copy()
        print("Created new merged dataset...")

    # Save merged CSV
    merged_df.to_csv(merged_csv, index=False)
    print(f"Saved merged dataset to: {merged_csv}")

    print(f"Done! Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--out", default="store", help="Output folder to save merged data")
    args = parser.parse_args()

    saveDataset(args.csv, args.out)
