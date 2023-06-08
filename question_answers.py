from config import *
import pinecone
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import generate_index
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.memory import ConversationBufferMemory

#from create_chunks import VectorDB
index_path = "index/faiss_pdf.index"

chat_history = []

def answer_question_from_local_index(query="What are transformers?"):
    try:
        #global db
        print("from local index")
        print(query)
        print(chat_history)
        embeddings = OpenAIEmbeddings()
        index = FAISS.load_local(index_path, embeddings)

        #When Index data is stored in a global var called db
        #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), generate_index.db.as_retriever())

        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), index.as_retriever())

        result = qa({"question": query, "chat_history": chat_history})
        print(result['answer'])
        chat_history.append((query, result['answer']))
        return result
    except Exception as e:
        print(e)
        return str(e)


def answer_question_from_external_db(query="What are transformers?"):
    try:
        #global db
        print("from pinecone index")
        print(query)
        print(chat_history)
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )
        index_name = PINECONE_INDEX
        # if not index_name in pinecone.list_indexes():
        #     print(pinecone.list_indexes())
        #     raise KeyError(f"Index '{index_name}' does not exist.")
        # index = pinecone.Index(index_name)

        embeddings = OpenAIEmbeddings()

        # if you already have an index, you can load it like this
        doc_db = Pinecone.from_existing_index(index_name, embeddings)

        #We can now create a memory object, which is neccessary to track the inputs/outputs and hold a conversation.
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        #We now initialize the ConversationalRetrievalChain
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), doc_db.as_retriever(), memory=memory)

        result = qa({"question": query})

        print(result['answer'])
        chat_history.append((query, result['answer']))
        return result
    except Exception as e:
        print(e)
        return str(e)