# Text-Scrapping-And-RAG

# 🔍 AI-Powered Web Scraping & Q&A System
![image](https://github.com/user-attachments/assets/ec38195c-ba19-4ff9-b4da-a0df7b7ebfdb)

This project is a **Streamlit-based  Q&A system** that:
- Scrapes text from a given URL.
- Generates embeddings using **Sentence Transformers**.
- Stores & retrieves data efficiently with **Pinecone**.
- Uses **Google Gemini API** for intelligent responses.
- Provides an interactive UI for seamless user experience.

---

## 📌 **Project Overview**

### **1️⃣ Web Scraping & Cleaning**
- Extracts **textual content** from a given webpage.
- Cleans unnecessary elements such as:
  - **Citations, URLs, extra spaces, and HTML tags**.
- Uses **BeautifulSoup & Regex** for preprocessing.

### **2️⃣ Text Embedding & Storage**
- Converts the cleaned text into **semantic embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
- **Splits** the text into chunks (to handle large data).
- Stores embeddings in **Pinecone**, a vector database.

### **3️⃣ Retrieval & Question-Answering**
- Takes a **user query** as input.
- Retrieves **top 3 most relevant text chunks** from Pinecone.
- Passes the retrieved context to **Google Gemini AI**.
- **Generates AI-based responses** for user queries.

---

## ⚙️ **How It Works**





### **1️⃣ Install Dependencies**
Make sure you have the required Python packages installed:
```sh
pip install streamlit requests beautifulsoup4 sentence-transformers pinecone-client google-generativeai
