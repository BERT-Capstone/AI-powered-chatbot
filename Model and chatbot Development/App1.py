from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import pipeline
import streamlit as st


# Load and preprocess PDF data
FILE_PATHS = ["./documents/migration_development_brief_38_june_2023_0.pdf"]
all_texts = []

for file_path in FILE_PATHS:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    texts = [page.page_content for page in pages]  # Extract text from each page
    all_texts.extend(texts)

# Combine all extracted texts into a single context string
context = " ".join(all_texts)

# Load pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to answer questions
def answer_question(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Streamlit app
    
st.title("Remittance Chatbot")

st.write("This app allows you to ask questions on remittance")

prompt = st.chat_input("Ask a question about remittance:")


if prompt:
    cont = context
    if context:
        st.write(f"Message by user: {prompt}")
        answer = answer_question(prompt, context)
        st.write("Answer:", answer)
    else:
        st.write("Cannot answer your question")
else:
    st.write("Please enter a question.")