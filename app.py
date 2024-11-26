import gradio as gr
from clearml import Task, PipelineDecorator
import pymongo
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
from datetime import datetime
import uuid


@PipelineDecorator.component(
    return_values=['documents'],
    cache=True
)
def extract_ros2_docs():
    repo_urls = [
        "https://docs.ros.org/en/humble/",
        "https://docs.nav2.org/",
        "https://moveit.picknik.ai/main/index.html",
        # "https://gazebosim.org/docs"
    ]

    documents = []
    # Document extraction logic here
    for url in repo_urls:
        response = requests.get(url)
        if response.status_code == 200:
            print(f'Successfully received response from {url}')
            print(f'{response.text}')
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
    return_values=['vectors'],
    cache=True
)
def create_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = []

    # First document to get embedding size
    sample_embedding = model.encode(documents[0]['content'])
    vector_size = len(sample_embedding)

    # Create collection with proper vector configuration
    qdrant = QdrantClient("localhost", port=6333)
    try:
        # Try to get collection to check if it exists
        qdrant.get_collection('ros2_docs')
    except:
        # Create collection with proper vector configuration
        qdrant.create_collection(
            collection_name='ros2_docs',
            vectors_config={  # Named vector configuration
                'size': vector_size,
                'distance': 'Cosine'
            }
        )

    # Process all documents
    for doc in documents:
        embedding = model.encode(doc['content'])
        vectors.append({
            'id': str(uuid.uuid4()),
            'vector': embedding.tolist(),  # Use the default vector name
            'payload': {
                'content': doc['content'],
                'source': str(doc.get('source', '')),
                'timestamp': str(doc.get('timestamp', '')),
                'mongodb_id': str(doc['_id'])
            }
        })

    # Upsert vectors in batches
    qdrant.upsert(
        collection_name='ros2_docs',
        points=vectors
    )

    return vectors


@PipelineDecorator.pipeline(
    name='ROS2_RAG_Pipeline',
    project='ROS2_RAG'
)
def pipeline_controller():
    # Execute pipeline steps
    documents = extract_ros2_docs()
    vectors = create_embeddings(documents)
    return vectors


class RAGSystem:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve_context(self, query, top_k=3):
        query_vector = self.embedder.encode(query)
        results = self.qdrant.search(
            collection_name="ros2_docs",
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        return [hit.payload['content'] for hit in results]

    def generate_response(self, query):
        context = self.retrieve_context(query)
        prompt = f"Context: {' '.join(context)}\nQuery: {query}"
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": "llama2", "prompt": prompt}
        )
        return response.json()['response']


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
    Task.init(project_name="ROS2_RAG", task_name="RAG_Pipeline")
    PipelineDecorator.run_locally()
    pipeline_controller()

    # Then start the Gradio interface
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
