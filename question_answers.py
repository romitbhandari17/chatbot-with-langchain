from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import create_chunks
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from create_chunks import VectorDB
index_path = "index/faiss_2.index"

chat_history = []

def answer_question_from_embeddings(query="Which are the best universities to study in london?"):
    try:
        #global db
        print(query)
        print(chat_history)
        embeddings = OpenAIEmbeddings()
        index = FAISS.load_local(index_path, embeddings)

        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), index.as_retriever())

        result = qa({"question": query, "chat_history": chat_history})
        print(result['answer'])
        chat_history.append((query, result['answer']))
        return result
    except Exception as e:
        print(e)
        return ""

    
    return result
