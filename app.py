import os
import pandas as pd
import spacy
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from evaluate_metrics import calculate_context_recall

# Load .env secrets
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Functions
def mask_pii(df):
    df = df.copy()
    if "email" in df.columns:
        df["email"] = df["email"].apply(lambda x: re.sub(r'[\\w.-]+@[\\w.-]+\\.\\w+', '[MASKED_EMAIL]', str(x)))
    if "customer_name" in df.columns:
        df["customer_name"] = "[MASKED_NAME]"
    return df

def chunk_dataframe(df, chunk_size=10):
    return [df.iloc[i:i+chunk_size].to_string(index=False) for i in range(0, len(df), chunk_size)]

# Streamlit App
st.set_page_config(page_title="E-Commerce Sales Agent", layout="wide")
st.title(" E-Commerce Sales Chatbot (RAG + LLM + spaCy)")

# Upload
file = st.file_uploader("Upload sales CSV", type="csv")
apply_masking = st.checkbox(" Mask PII")

if file:
    df = pd.read_csv(file)
    if apply_masking:
        df = mask_pii(df)

    chunks = chunk_dataframe(df)
    docs = [Document(page_content=c) for c in chunks]

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)

    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input(" Ask a sales-related question")

    if query:
        answer = qa.run(query)
        st.markdown("###  Answer:")
        st.success(answer)

        # Optional: Show chunks used
        st.markdown("### Retrieved Chunks")
        retrieved = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(retrieved):
            st.code(f"[Chunk {i+1}]\n{doc.page_content}")

        recall = calculate_context_recall([doc.page_content for doc in retrieved], answer)
        st.info(f"Context Recall Score: {recall:.2f}")

else:
    st.warning("Please upload a sales CSV file to begin.")
