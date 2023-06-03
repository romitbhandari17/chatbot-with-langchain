import os
import tiktoken  # for counting tokens
#from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from flask import jsonify

os.environ["OPENAI_API_KEY"] = "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"


# You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)

# Simple method - Split by pages 
'''loader = PyPDFLoader("attention-is-all-u-need.pdf")
pages = loader.load_and_split()
print(pages[0])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages'''

# Advanced method - Split by chunk

# Step 1: Convert PDF to text

#doc = textract.process("data/attention-is-all-u-need.pdf", method='pdfminer')
#doc = textract.process("data/wur_ranking_summary.csv")

# Step 2: Save to .txt and reopen (helps prevent issues)
#with open('attention_is_all_you_need.txt', 'w') as f:
#    f.write(doc.decode('utf-8'))
        

def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    #print("Start of num_tokens func")
    #encoding = tiktoken.encoding_for_model(model)
    embedding_encoding = "cl100k_base"
    encoding = tiktoken.get_encoding(embedding_encoding)
    return len(encoding.encode(text))


def create_chunks_and_embeddings():
    try:
        global vector_index
        with open('data/attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        #decoded_data = data.decode('utf-8')

        # Step 3: Create function to count tokens
        '''tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))'''


        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 512,
            chunk_overlap  = 24,
            length_function = num_tokens,
        )

        print("before chunk creation")
        chunks = text_splitter.create_documents([text])
        print("after chunk creation")
        # Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
        type(chunks[0])

        # Quick data visualization to ensure chunking was successful

        # Create a list of token counts
        #token_counts = [self.num_tokens(chunk.page_content) for chunk in chunks]

        # Create a DataFrame from the token counts
        #df = pd.DataFrame({'Token Count': token_counts})

        # Create a histogram of the token count distribution
        #df.hist(bins=40, )

        # Show the plot
        #plt.show()

        """# 2. Embed text and store embeddings"""

        # Get embedding model
        embeddings = OpenAIEmbeddings()

        print("before vector")
        # Create vector database
        vector_index = FAISS.from_documents(chunks, embeddings)
        #FAISS.write_index(vector_index, 'my_faiss_index.index')
        print("after vector")
        """# 5. Create chatbot with chat memory (OPTIONAL) """

        # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
        #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

        return jsonify({"embeddings created": True})
        #return "embeddings created"
    except Exception as e:
        print(e)
        return ""