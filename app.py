from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import gradio as gr
from clearml import Task, PipelineDecorator
import pymongo
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import requests
from datetime import datetime
import uuid
import json
from utils import crawl_website
import json

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ros2_rag"]
collection = db["raw_docs"]

# @PipelineDecorator.component(
#     return_values=['documents'],
#     cache=False
# )


def extract_ros2_subdomain_docs():
    f = open('subdomain_links.json')
    repo_urls = json.load(f)['links']

    documents = []
    try:
        # Crawl each website
        for url in repo_urls:
            print(f"\nCrawling {url}")
            data_list = crawl_website(url, 50)
            if data_list:
                try:
                    collection.insert_many(data_list)
                    documents.extend(data_list)
                    print(f"Stored {len(data_list)} pages from {url}")
                except Exception as e:
                    print(f"Error storing data from {url}: {str(e)}")
    except Exception as e:
        print(f"An error occurred while crawling: {str(e)}")
    finally:
        return documents


# @PipelineDecorator.component(
#     return_values=['vectors'],
#     cache=True
# )
def create_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = []

    # First document to get embedding size
    # sample_embedding = model.encode(documents[0]['content'])
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Vector Size ===> {vector_size}")

    # Create collection with proper vector configuration
    qdrant = QdrantClient("localhost", port=6333)
    try:
        # Try to get collection to check if it exists
        qdrant.get_collection('ros2_docs')
        print("Collection ros2_docs exists")
    except:
        # Create collection with proper vector configuration
        qdrant.create_collection(
            collection_name='ros2_docs',
            vectors_config=models.VectorParams(  # Named vector configuration
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print("Created new collection: ros2_docs!")

    # Process all documents
    for doc in documents:
        # embedding = model.encode(doc['content'])
        embedding = model.encode(doc['text_content'])
        vectors.append(models.PointStruct(
            id=f"{str(uuid.uuid4())}",
            vector=embedding.tolist(),  # Use the default vector name
            payload={
                "url": doc['url'],
                "text_content": doc['text_content']
            }
        ))

    # Upsert vectors in batches
    qdrant.upload_points(
        collection_name='ros2_docs',
        points=vectors
    )

    return vectors


# @PipelineDecorator.pipeline(
#     name='ROS2_RAG_Pipeline',
#     project='ROS2_RAG',
# )
def pipeline_controller():
    # Execute pipeline steps
    documents = extract_ros2_subdomain_docs()
    vectors = create_embeddings(documents)
    return vectors


class RAGSystem:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve_context(self, query, top_k=3):
        query_vector = self.model.encode(query)
        results = self.qdrant.search(
            collection_name="ros2_docs",
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        print(results)
        return [hit.payload['text_content'] for hit in results]

    def generate_response(self, common_question, custom_question):
        # Prefer the custom question if provided
        query = custom_question if custom_question else common_question
        context = self.retrieve_context(query)
        print(f'CONTEXT RETRIEVED =====> \n {" ".join(context)} \n')
        prompt = f"Context: {' '.join(context)}\nQuery: {query}"
        payload = {"model": "llama3.2", "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            headers=headers,
            stream=True
        )
        print(f"RESPONSE RECEIVED !!! ===> \n {response.text}")
        # return response.json()['response']
        response.raise_for_status()  # Ensure request was successful

        # Process streamed JSON responses
        full_response = ""  # To store the complete response
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    # Parse the JSON object
                    data = json.loads(chunk)
                    # Extract the "response" text
                    partial_text = data.get("response", "")
                    full_response += partial_text  # Append to the complete response
                    yield full_response  # Yield updated response for streaming
                except json.JSONDecodeError:
                    print("Failed to decode JSON chunk:", chunk)
        return full_response


def create_gradio_interface():
    rag = RAGSystem()

    demo = gr.Interface(
        fn=rag.generate_response,
        inputs=[
            gr.Dropdown(
                choices=[
                    "Tell me how can I navigate to a specific pose - include replanning aspects in your answer.",
                    "Can you provide me with code for this task?"
                ],
                label="Common Questions"
            ),
            gr.Textbox(label="Or ask your own question")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="ROS2 Navigation Assistant"
    )
    return demo


if __name__ == "__main__":
    # First run the pipeline
    # Task.init(project_name="ROS2_RAG", task_name="RAG_Pipeline")
    # PipelineDecorator.run_locally()
    pipeline_controller()

    # Then start the Gradio interface
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
