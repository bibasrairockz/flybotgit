from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI


from pinecone import Pinecone

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
# pinecone.init(api_key = PINECONE_API_KEY,
#               environment = PINECONE_API_ENV)
pc = Pinecone(api_key=PINECONE_API_KEY)
openai.api_key = OPENAI_API_KEY

index_name="starter-index"
index = pc.Index(index_name)
client = OpenAI(api_key = OPENAI_API_KEY)


#Loading the index
# docsearch=Pinecone.from_existing_index(index_name, embeddings)
from langchain.vectorstores import Pinecone
text_field = "text"
vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)



PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})
gptModel = ChatOpenAI(
    model = "gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0
)
# print(os.getcwd())

qa = RetrievalQA.from_chain_type(
    llm=gptModel, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8000, debug= True)
 
