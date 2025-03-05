import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

#  Configure API Keys
GOOGLE_API_KEY = "google_api_key"  # Replace with your actual API key
PINECONE_API_KEY = "pine_cone_key"  # Replace with your Pinecone API key
PINECONE_ENV = "us-east1-gcp"  # Replace with your Pinecone environment
INDEX_NAME = "my-index"  # Pinecone index name

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#  Scraping Function
def scrape_and_clean(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[\d+\]', '', text)  # Remove references
    return text.strip()

#  Store Embeddings in Pinecone
def store_embeddings(text):
    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    vectors = [(f"text_{i}", embedding_model.encode(chunk).tolist(), {"text": chunk}) for i, chunk in enumerate(chunks)]
    index.upsert(vectors=vectors)
    return len(vectors)

#  Retrieve Relevant Chunks
def retrieve_relevant_text(query):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return " ".join([match.get("metadata", {}).get("text", "") for match in results.get("matches", [])])

# Generate AI Response
def generate_ai_response(query, retrieved_docs):
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(f"Use this context to answer the query: {retrieved_docs}. Query: {query}")
        return response.text if response else " No response generated."
    except Exception as e:
        return f" Error generating response: {str(e)}"

# Streamlit UI
st.title("Text Scraping and RAG")
st.write("This app scrapes a webpage, stores embeddings in Pinecone, and generates AI responses using Gemini.")

# Input: URL & Process Button
url = st.text_input(" Enter a webpage URL:", "https://en.wikipedia.org/wiki/Natural_language_processing")
if st.button("Scrape & Store"):
    text = scrape_and_clean(url)
    num_chunks = store_embeddings(text)
    st.success(f"Scraped & stored {num_chunks} chunks in Pinecone!")

# Input: User Query
query = st.text_input(" Ask a question:")
if query:
    st.write("Retrieving relevant documents...")
    retrieved_text = retrieve_relevant_text(query)

    st.write(" **Top Retrieved Documents:**")
    for i, chunk in enumerate(retrieved_text.split(". ")[:3]):
        st.write(f"**{i+1}.** {chunk}...")

    st.write(" **AI Response:**")
    ai_response = generate_ai_response(query, retrieved_text)
    st.write(ai_response)

