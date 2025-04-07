from flask import Flask, render_template, jsonify,request

from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os 



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = "pcsk_bkA7R_QrV1CyjrzUqiiVJrbb4cwySPnt6PsVgn2QfHq6ys8eWyAy3gFK5vGgtYLRCsT3x"
PINECONE_API_ENV = "us-east-1"

embeddings = download_hugging_face_embeddings()

#Initializing pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# If we already have an index we can load it like this
docsearch = PineconeVectorStore(index_name,embeddings)

PROMPT = PromptTemplate(template=Prompt_template, input_variables=["context","question"])


chain_type_kwargs = {"prompt":PROMPT}

# Load LLM
llm = CTransformers(
    model="E:/5)End-To-End-Medical-Chatbot/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.8
    }
)



# Create RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


# Creating default route
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response:", result["result"])
    return str(result["result"])



if __name__== "__main__":
    app.run(debug=True)








