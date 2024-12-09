from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from pymongo import MongoClient
import uuid
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from clearml import Task, PipelineDecorator
import re

load_dotenv()
nltk.download('punkt')


class DataFeaturizer:
    def __init__(self):
        # MongoDB setup
        self.mongo_client = MongoClient(os.getenv('MONGO_URI'))
        self.db = self.mongo_client['ros2_rag']

        # Qdrant setup
        self.qdrant = QdrantClient("localhost", port=int(os.getenv('QDRANT_PORT')))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Collection names
        self.qdrant_collection = "ros2_vectors"

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def chunk_text(self, text, max_chunk_size=512):
        """Split text into chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_vectors(self, chunks, metadata):
        """Create vectors from text chunks"""
        vectors = []
        for chunk in chunks:
            embedding = self.model.encode(chunk)
            vectors.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": metadata.get('source', 'unknown'),
                    "url": metadata.get('url', ''),
                    "type": metadata.get('type', 'unknown')
                }
            ))
        return vectors

    def initialize_qdrant(self):
        """Initialize Qdrant collection"""
        vector_size = self.model.get_sentence_embedding_dimension()
        try:
            self.qdrant.get_collection(self.qdrant_collection)
            print(f"Collection {self.qdrant_collection} exists")
        except:
            self.qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {self.qdrant_collection}")

    def process_github_data(self):
        """Process GitHub repository data"""
        collection = self.db['github_repo']
        vectors = []

        for doc in collection.find():
            if doc.get('content'):
                print(f"Processing https://github.com/{doc['owner']}/{doc['repo']}/blob/main/{doc['path']}")
                clean_content = self.clean_text(doc['content'])
                chunks = self.chunk_text(clean_content)
                metadata = {
                    'source': 'github',
                    'url': f"https://github.com/{doc['owner']}/{doc['repo']}/blob/main/{doc['path']}",
                    'type': 'documentation'
                }
                vectors.extend(self.create_vectors(chunks, metadata))

        return vectors

    def process_web_docs(self):
        """Process web documentation"""
        collection = self.db['raw_docs']
        vectors = []

        for doc in collection.find():
            if doc.get('text_content'):
                print(f"Processing website data: {doc.get('url', '--No url found--')}")
                clean_content = self.clean_text(doc['text_content'])
                chunks = self.chunk_text(clean_content)
                metadata = {
                    'source': 'web',
                    'url': doc.get('url', ''),
                    'type': 'documentation'
                }
                vectors.extend(self.create_vectors(chunks, metadata))

        return vectors

    def process_youtube_data(self):
        """Process YouTube transcripts"""
        collection = self.db['youtube_transcripts']
        vectors = []

        for doc in collection.find():
            if doc.get('transcript'):
                print(f"Processing transcript from video: {doc.get('video_id')}")
                clean_content = self.clean_text(doc['transcript'])
                chunks = self.chunk_text(clean_content)
                metadata = {
                    'source': 'youtube',
                    'url': f"https://youtube.com/watch?v={doc['video_id']}",
                    'type': 'video_transcript'
                }
                vectors.extend(self.create_vectors(chunks, metadata))

        return vectors

    def process_all_data(self):
        """Process all data sources and upload to Qdrant"""
        self.initialize_qdrant()

        # Process each data source
        all_vectors = []
        all_vectors.extend(self.process_github_data())
        all_vectors.extend(self.process_web_docs())
        all_vectors.extend(self.process_youtube_data())

        # Upload vectors in batches
        batch_size = 100
        for i in range(0, len(all_vectors), batch_size):
            batch = all_vectors[i:i + batch_size]
            self.qdrant.upload_points(
                collection_name=self.qdrant_collection,
                points=batch
            )
            print(f"Uploaded batch {i//batch_size + 1}/{len(all_vectors)//batch_size + 1}")

        return len(all_vectors)


@PipelineDecorator.component(return_values=['total_vectors'])
def run_featurizer():
    featurizer = DataFeaturizer()
    total_vectors = featurizer.process_all_data()
    print(f"Total vectors created: {total_vectors}")
    return total_vectors


if __name__ == "__main__":
    # task = Task.init(project_name='ROS2_RAG', task_name='Data Featurization')
    run_featurizer()
