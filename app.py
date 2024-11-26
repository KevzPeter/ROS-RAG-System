import gradio as gr
from clearml import Task, PipelineDecorator
import pymongo
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import requests
from datetime import datetime
import uuid
import json


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
            # print(f'{response.text}')
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
    # sample_embedding = model.encode(documents[0]['content'])
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Vector Size ===> {vector_size}")

    # Create collection with proper vector configuration
    qdrant = QdrantClient("localhost", port=6333)
    try:
        # Try to get collection to check if it exists
        qdrant.get_collection('ros2_docs')
        print("COLLECTION ALREADY EXISTS!!!")
    except:
        # Create collection with proper vector configuration
        qdrant.create_collection(
            collection_name='ros2_docs',
            vectors_config=models.VectorParams(  # Named vector configuration
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print("Created New Collection ros2_docs!")

    # Process all documents
    for doc in documents:
        # embedding = model.encode(doc['content'])
        embedding = model.encode(doc['description'])
        vectors.append(models.PointStruct(
            id=f"{str(uuid.uuid4())}",
            vector=embedding.tolist(),  # Use the default vector name
            # 'vector': {},
            # payload={
            #     'content': doc['content'],
            #     'source': str(doc.get('source', '')),
            #     'timestamp': str(doc.get('timestamp', '')),
            #     'mongodb_id': str(doc['_id'])
            # }
            payload=doc
        ))

    # Upsert vectors in batches
    qdrant.upload_points(
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
    # documents = extract_ros2_docs()
    documents = [
        {
            "name": "The Time Machine",
            "description": "A man travels through time and witnesses the evolution of humanity.",
            "author": "H.G. Wells",
            "year": 1895,
        },
        {
            "name": "Ender's Game",
            "description": "A young boy is trained to become a military leader in a war against an alien race.",
            "author": "Orson Scott Card",
            "year": 1985,
        },
        {
            "name": "Brave New World",
            "description": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
            "author": "Aldous Huxley",
            "year": 1932,
        },
        {
            "name": "The Hitchhiker's Guide to the Galaxy",
            "description": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
            "author": "Douglas Adams",
            "year": 1979,
        },
        {
            "name": "Dune",
            "description": "A desert planet is the site of political intrigue and power struggles.",
            "author": "Frank Herbert",
            "year": 1965,
        },
        {
            "name": "Foundation",
            "description": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
            "author": "Isaac Asimov",
            "year": 1951,
        },
        {
            "name": "Snow Crash",
            "description": "A futuristic world where the internet has evolved into a virtual reality metaverse.",
            "author": "Neal Stephenson",
            "year": 1992,
        },
        {
            "name": "Neuromancer",
            "description": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
            "author": "William Gibson",
            "year": 1984,
        },
        {
            "name": "The War of the Worlds",
            "description": "A Martian invasion of Earth throws humanity into chaos.",
            "author": "H.G. Wells",
            "year": 1898,
        },
        {
            "name": "The Hunger Games",
            "description": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
            "author": "Suzanne Collins",
            "year": 2008,
        },
        {
            "name": "The Andromeda Strain",
            "description": "A deadly virus from outer space threatens to wipe out humanity.",
            "author": "Michael Crichton",
            "year": 1969,
        },
        {
            "name": "The Left Hand of Darkness",
            "description": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
            "author": "Ursula K. Le Guin",
            "year": 1969,
        },
        {
            "name": "The Three-Body Problem",
            "description": "Humans encounter an alien civilization that lives in a dying system.",
            "author": "Liu Cixin",
            "year": 2008,
        },
    ]
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
        return [hit.payload['description'] for hit in results]

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
    Task.init(project_name="ROS2_RAG", task_name="RAG_Pipeline")
    PipelineDecorator.run_locally()
    pipeline_controller()

    # Then start the Gradio interface
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
