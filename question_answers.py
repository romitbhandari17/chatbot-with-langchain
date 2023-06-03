from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import create_chunks
from langchain.vectorstores import FAISS
#from create_chunks import VectorDB

chat_history = []

def answer_question_from_embeddings(query="Which are the best universities to study in london?"):
    try:
        #global db
        print(query)
        print(chat_history)
        #db_instance = VectorDB.get_instance()
        #faiss_index = FAISS.read_index('my_faiss_index.index')
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), create_chunks.vector_index.as_retriever())
        #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), faiss_index.as_retriever())

        result = qa({"question": query, "chat_history": chat_history})
        print(result['answer'])
        chat_history.append((query, result['answer']))
        return result
    except Exception as e:
        print(e)
        return ""

    
    return result
