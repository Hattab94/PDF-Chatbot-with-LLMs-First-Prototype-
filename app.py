import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

st.title("PDF Chatbot with LLMs")

uploaded_file = st.file_uploader("C:/Users/acer/Desktop/1745767076368.pdf", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(texts, embeddings)

    query = st.text_input("Ask something about the PDF:")
    if query:
        docs = db.similarity_search(query)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        st.write("Answer:", answer)