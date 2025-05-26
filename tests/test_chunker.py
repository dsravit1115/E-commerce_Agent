from agent.chunker import chunk_dataframe, mask_pii
import pandas as pd

def test_chunking():
    df = pd.read_csv("data/sales_data.csv")
    chunks = chunk_dataframe(df, chunk_size=2)
    assert len(chunks) > 0

def test_masking():
    df = pd.read_csv("data/sales_data.csv")
    masked_df = mask_pii(df)
    assert "[MASKED_EMAIL]" in masked_df["email"].iloc[0] or "[MASKED_NAME]" in masked_df["customer_name"].iloc[0]
