from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('.env')

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(os.getenv('MONGO_URI'))
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader('./data', glob="./*.txt", show_progress=True)
data = loader.load()

# # Define the OpenAI Embedding Model we want to use for the source data
# # The embedding model is different from the language generation model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

# # Initialize the VectorStore, and
# # vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)
