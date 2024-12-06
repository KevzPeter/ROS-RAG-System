import datetime
from clearml import Task, PipelineDecorator
import requests
import pymongo
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from clearml import Task

Task.add_requirements("requirements.txt")
Task.set_base_docker("python:3.9-slim")

Task.init(project_name="ROS2_RAG", task_name="ETL_Pipeline")
task = Task.current_task()
task.set_parameters({
    "batch_size": 32,
    "embedding_model": "all-MiniLM-L6-v2",
    "mongodb_url": "mongodb://localhost:27017/",
    "qdrant_url": "localhost:6333"
})

@PipelineDecorator.component(
    return_values=['raw_docs'],
    cache=True
)
def extract_github_docs(repo_urls):
    documents = []
    for url in repo_urls:
        response = requests.get(url)
        if response.status_code == 200:
            documents.append({
                'content': response.text,
                'source': url,
                'timestamp': datetime.now()
            })
    
    # Store in MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ros2_rag"]
    collection = db["raw_docs"]
    collection.insert_many(documents)
    
    return documents

@PipelineDecorator.component(
    return_values=['embeddings'],
    cache=True
)
def create_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    
    for doc in documents:
        vector = model.encode(doc['content'])
        embeddings.append({
            'vector': vector,
            'payload': doc
        })
    
    client = QdrantClient("localhost", port=6333)
    client.upsert(
        collection_name="ros2_docs",
        points=embeddings
    )
    
    return embeddings

@PipelineDecorator.pipeline(
    name='RAG_ETL_Pipeline',
    project='ROS2_RAG'
)
def pipeline_controller():
    # Define source URLs
    urls = [
        # "https://docs.ros.org/en/humble/",
        "https://docs.nav2.org/behavior_trees/trees/nav_to_pose_with_consistent_replanning_and_if_path_becomes_invalid.html"
    ]
    
    # Execute pipeline steps
    raw_docs = extract_github_docs(urls)
    embeddings = create_embeddings(raw_docs)
    
    return embeddings

if __name__ == "__main__":
    Task.init(project_name="ROS2_RAG", task_name="ETL_Pipeline")
    pipeline_controller()