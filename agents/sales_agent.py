import pandas as pd

def calculate_sales_summary(_):
    df = pd.read_csv("data/sales_data.csv")
    summary = df.groupby("region")["amount"].sum().reset_index()
    return summary.to_string(index=False)
