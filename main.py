import os
import pandas as pd
import spacy
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from agent.sales_agent import calculate_sales_summary
from agent.chunker import chunk_dataframe, mask_pii

# Load secrets
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load and mask sales data
df = pd.read_csv("data/sales_data.csv")
df = mask_pii(df)

# Chunk data
chunks = chunk_dataframe(df)
documents = [Document(page_content=chunk) for chunk in chunks]

# Embedding and vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding)

# Retrieval QA setup
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Agent with tool
tools = [Tool(name="SalesSummary", func=calculate_sales_summary, description="Summarizes sales data")]
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

query = "Give me total sales by region for last month"
print("Agent Output:", agent.run(query))
