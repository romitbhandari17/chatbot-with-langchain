import os
import tiktoken  # for counting tokens
from config import *
import pinecone
#from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores import Pinecone
from typing import List
from flask import jsonify

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

text_path = "data/attention_is_all_you_need.txt"
index_path = "index/faiss_pdf.index"

def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    #print("Start of num_tokens func")
    #encoding = tiktoken.encoding_for_model(model)
    embedding_encoding = "cl100k_base"
    encoding = tiktoken.get_encoding(embedding_encoding)
    return len(encoding.encode(text))

def load_documents() -> List:
    loader = TextLoader(text_path,encoding='utf-8')
    return loader.load()

def load_raw_documents() -> List:
    loader = DirectoryLoader('./docs/pdf', glob="*.pdf")

    return loader.load()


def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


def generate_index_store_locally():
    try:
        print("before load doc")
        #sources = load_documents()
        sources = load_raw_documents()
        # with open('data/attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
        #     text = f.read()

        print("before split chunks")
        chunks = split_chunks(sources)
        print("after split chunks")

        # we use the openAI embedding model
        embeddings = OpenAIEmbeddings()
        print("after openai embeddings")

        vectorstore = generate_index(chunks, embeddings)
        print("after generate index")
        #vectorstore.save_local("full_sotu_index")
        vectorstore.save_local(index_path)
        print("after index save locally")

        return jsonify({"Index generated and stored internally": True})
    except Exception as e:
        print(e)
        return ""
    

def generate_index_store_externally():
    try:
        print("before load doc")
        #sources = load_documents()
        sources = load_raw_documents()

        print("before split chunks")
        chunks = split_chunks(sources)
        print("after split chunks")

        # we use the openAI embedding model
        embeddings = OpenAIEmbeddings()
        print("after openai embeddings")
        
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

        doc_db = Pinecone.from_documents(
            chunks, 
            embeddings, 
            index_name=PINECONE_INDEX
        )

        # We can now search for relevant documents in that database using the cosine similarity metric

        # query = "What were the most important events for Google in 2021?"
        # search_docs = doc_db.similarity_search(query)
        # search_docs
        print("after index save in pinecone")

        return jsonify({"Index generated and stored externally": True})
    except Exception as e:
        print(e)
        return ""