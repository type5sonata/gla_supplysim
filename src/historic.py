"""Loading the historic applications/approvals/starts/completions series."""

import os
import pandas as pd

def load_historic_data():
    """Load historic data from CSV file"""
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'historic.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Rename the quarter column for consistency
        df = df.rename(columns={'Year + Q': 'Quarter'})
        # Clean up any comma-formatted numbers
        for col in ['Applications', 'Approvals', 'Starts', 'Completions']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float).astype(int)
        # Remove duplicate quarters, keeping the last occurrence
        df = df.drop_duplicates(subset=['Quarter'], keep='last')
        return df
    return None

def filter_historic_data(historic_df, start_year, start_quarter):
    """Filter historic data to only include data before the start date"""
    if historic_df is None:
        return None

    filtered_rows = []
    for _, row in historic_df.iterrows():
        quarter_str = row['Quarter']
        # Parse the quarter string (e.g., "2024 Q1")
        parts = quarter_str.split()
        year = int(parts[0])
        quarter = int(parts[1][1])

        # Include only data before the start date
        if year < start_year or (year == start_year and quarter < start_quarter):
            filtered_rows.append(row)

    if filtered_rows:
        return pd.DataFrame(filtered_rows)
    return None

