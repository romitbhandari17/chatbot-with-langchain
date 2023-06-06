# MBAGPT: Chatting with Multiple Data Sources

This is a chat application that leverages the power of Langchain conversational retrieval chain to provide conversational responses with access to a data source either PDF or CSV. Typically, following steps are followed:
- The Input data source is first converted to text file, divided into chunks using langchain Recursive Character TextSplitter.
- Then a vector index generated using OpenAI embeddings model and the chunks created. 
- Finally, the vector index is stored in a local file or variable.
- When a question is asked, using Conversational Retrieval Chain mechanism of Langchain, answers are provided using the index created in the last step.


## Features

- Chat interface for interacting with the chatbot powered by Langchain and OpenAI APIs.
- Integration with the input data source for retrieving relevant information.
- Semantic search functionality to provide informative snippets from the databases.
- Answers based on persistent Chat memory.
- HTML templates for displaying chat history and messages.
- Persistence of embeddings using the FAISS vector store.
- OpenAI API key integration for authentication.

## Installation

1. Clone the repository:

```
git clone https://github.com/romitbhandari17/chatbot-with-langchain.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Set up your credentials:

- Sign up on the OpenAI website and obtain an API key.
- Create a new file called "config.yaml" in the root folder.
- Set your OpenAI API key (required) and pinecone creds (optional) in the config.yaml file with Key 'OPENAI_API_KEY'.
- Update the code in the app to use the correct method for accessing the API key.

4. Add the PDF/CSV input file in a /data folder created under project root. Convert the PDF or CSV file input to a text file seperately using Collab or other means and add it under the same /data folder.

5. Run the application:

```
python app.py
```

6. Give the Index path in create_chunks.py and question_answers.py file.

7. Create the vector index by calling the GET API with end point /create_embeddings from browser or POSTMAN. This call will create the vector index using the raw txt file and store it in the local file.

8. Start using the chatbot with endpoint /chatbot.

## Usage

1. Access the application by navigating to `http://localhost:8080/chatbot` in your web browser.

2. Enter your prompt in the input box and press Enter.

3. The chatbot will process your prompt and provide a response based on the available data sources.

4. The chat history will be displayed on the screen, showing both user and assistant messages.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Implement your changes and ensure that the code passes all tests.

4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License.
