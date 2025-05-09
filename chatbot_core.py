# chatbot_core.py: This module defines the core logic of the PDFchatbot.
# It builds a RAG pipeline using LangChain.

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain

def build_qa_chain(pdf_path="ucf_rules.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()[1:]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    llm = ChatOllama(model="mistral")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain