import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def mask_pii(df):
    df = df.copy()
    if "email" in df.columns:
        df["email"] = df["email"].apply(lambda x: re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[MASKED_EMAIL]', str(x)))
    if "customer_name" in df.columns:
        df["customer_name"] = df["customer_name"].apply(lambda x: "[MASKED_NAME]")
    return df

def chunk_dataframe(df, chunk_size=10):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].to_string(index=False)
        chunks.append(chunk)
    return chunks
