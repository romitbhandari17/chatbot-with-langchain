from __future__ import print_function
from config import *
import sys
import logging
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS, cross_origin
from flask import request

#from handle_file import handle_file
from create_chunks import create_chunks_and_embeddings
from question_answers import answer_question_from_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

vector_index = None

def create_app():
    #pinecone_index = load_pinecone_index()
    #tokenizer = tiktoken.get_encoding("gpt2")
    #session_id = str(uuid.uuid4().hex)
    app = Flask(__name__)
    #app.pinecone_index = pinecone_index
    #app.tokenizer = tokenizer
    #app.session_id = session_id
    # log session id
    #logging.info(f"session_id: {session_id}")
    #app.config["file_text_dict"] = {}
    with app.app_context():
        CORS(app, supports_credentials=True)

        #vector_db = VectorDB.get_instance()
        #create_embeddings_response  = create_chunks_and_embeddings()
        #print(create_embeddings_response)

    return app

app = create_app()

@app.route(f"/create_embeddings", methods=["GET"])
@cross_origin(supports_credentials=True)
def process_file():
    try:
        '''with open('create_embeddings_rankings.py', 'r') as file:
            code = file.read()
        
        exec(code)'''

        #vector_db = VectorDB.get_instance()
        create_embeddings_response = create_chunks_and_embeddings()
        return create_embeddings_response
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})
    

@app.route(f"/answer_question", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        
        '''show_chatbot_response = show_chatbot()
        return show_chatbot_response'''

        params = request.get_json()
        print(params)
        question = params["question"]

        answer_question_response = answer_question_from_embeddings(
            question)
        return answer_question_response
    except Exception as e:
        return str(e)
    

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"

@app.route('/chatbot', methods=['GET'])
def chatbot_ui():
    return render_template('chatbot_advanced.html')

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)