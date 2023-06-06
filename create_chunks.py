import os
import tiktoken  # for counting tokens
from config import *
#from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from typing import List
from flask import jsonify

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

text_path = "data/attention_is_all_you_need.txt"
index_path = "index/faiss_2.index"

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


def create_chunks_and_embeddings():
    try:
        print("before load doc")
        sources = load_documents()
        # with open('data/attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
        #     text = f.read()

        print("before split chunks")
        chunks = split_chunks(sources)
        print("after split chunks")
        # Get embedding model
        embeddings = OpenAIEmbeddings()
        print("after openai embeddings")
        vectorstore = generate_index(chunks, embeddings)
        print("after generate index")
        #vectorstore.save_local("full_sotu_index")
        vectorstore.save_local(index_path)
        print("after index save")

        return jsonify({"embeddings created": True})
        #return "embeddings created"
    except Exception as e:
        print(e)
        return ""