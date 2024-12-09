import tiktoken  # A library for counting tokens; works with LLMs like OpenAI's models
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
import json
from dotenv import load_dotenv
import os
import web_crawler
import youtube_ingester
import github_ingester
import featurizer
load_dotenv()

mongo_uri = os.getenv('MONGO_URI')
client = pymongo.MongoClient(mongo_uri)
db = client["ros2_rag"]
collection = db["raw_docs"]

qdrant_collection = "ros2_docs"


@PipelineDecorator.pipeline(
    name='ROS2_RAG_Pipeline',
    project='ROS2_RAG',
)
def pipeline_controller():
    # Run pipeline to crawl all 4 subdomain links
    web_crawler.crawl_subdomains()
    # Run pipeline to ingest github repositories
    github_ingester.run_github_ingester()
    # Run pipeline to ingest youtube video transcripts
    youtube_ingester.run_youtube_ingester()
    # Featurization: Clean, Chunk, Embed
    featurizer.run_featurizer()


class RAGSystem:
    def __init__(self, max_tokens=2048):
        self.qdrant = QdrantClient("localhost", port=os.getenv('QDRANT_PORT'))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.history = []  # Sliding window for conversation history
        self.max_tokens = max_tokens  # Maximum tokens for Llama3.2 (adjust as needed)

    def retrieve_context(self, query, top_k=3):
        query_vector = self.model.encode(query)
        results = self.qdrant.search(
            collection_name="ros2_docs",
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        print(results)
        return [hit.payload['text_content'] for hit in results]

    def count_tokens(self, text):
        """Estimate token count based on word count."""
        words = text.split()  # Split text into words
        return int(len(words) * 1.3)  # Estimate tokens as 1.3 tokens per word

    def trim_history(self, system_message, context, query):
        """
        Trim conversation history to fit within the token limit.
        """
        system_tokens = self.count_tokens(system_message)
        context_tokens = self.count_tokens(context)
        query_tokens = self.count_tokens(query)
        available_tokens = self.max_tokens - system_tokens - context_tokens - query_tokens

        trimmed_history = []
        total_tokens = 0

        # Add history from the most recent backwards, ensuring token count is within limits
        for user_message, assistant_response in reversed(self.history):
            entry = f"User: {user_message}\nAssistant: {assistant_response}\n"
            entry_tokens = self.count_tokens(entry)

            if total_tokens + entry_tokens > available_tokens:
                break  # Stop adding if we exceed the limit

            trimmed_history.insert(0, entry)  # Insert at the beginning
            total_tokens += entry_tokens

        return "\n".join(trimmed_history)

    def generate_response(self, common_question, custom_question):
        query = custom_question if custom_question else common_question
        context = " ".join(self.retrieve_context(query))

        # System message for context setting
        system_message = "You are an expert in ROS2 robotics. Provide concise and accurate answers to user queries."

        # Trim conversation history
        history_text = self.trim_history(system_message, context, query)

        # Construct the prompt
        prompt = (
            f"{system_message}\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Current Context: {context}\n"
            f"User: {query}\nAssistant:"
        )

        payload = {"model": "llama3.2", "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            headers=headers,
            stream=True
        )
        print(f"RESPONSE RECEIVED !!! ===> \n {response.text}")
        response.raise_for_status()  # Ensure request was successful

        # Process streamed JSON responses
        full_response = ""  # To store the complete response
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    # Parse the JSON object
                    data = json.loads(chunk)
                    partial_text = data.get("response", "")
                    full_response += partial_text  # Append to the complete response
                    yield full_response  # Yield updated response for streaming
                except json.JSONDecodeError:
                    print("Failed to decode JSON chunk:", chunk)

        # Append the current query and response to history
        self.history.append((query, full_response))
        return full_response


# @PipelineDecorator.component(return_values=['demo'], cache=False)
# def create_gradio_interface():
#     rag = RAGSystem()

#     demo = gr.Interface(
#         fn=rag.generate_response,
#         inputs=[
#             gr.Dropdown(
#                 choices=[
#                     "Tell me how can I navigate to a specific pose - include replanning aspects in your answer.",
#                     "Can you provide me with code for this task?"
#                 ],
#                 label="Common Questions"
#             ),
#             gr.Textbox(label="Or ask your own question")
#         ],
#         outputs=gr.Textbox(label="Answer"),
#         title="ROS2 Navigation Assistant"
#     )
#     return demo

def create_gradio_interface():
    rag = RAGSystem()

    def respond(message, history):
        response_generator = rag.generate_response(message, message)
        response = ""
        for chunk in response_generator:
            response = chunk
            yield response

    with gr.Blocks() as demo:
        chatbot = gr.ChatInterface(
            fn=respond,
            examples=[
                ["Tell me how can I navigate to a specific pose - include replanning aspects in your answer."],
                ["Can you provide me with code for this task?"]
            ],
            title="ROS2 Navigation Assistant",
            description="Ask questions about ROS2 navigation and get detailed answers with code examples when applicable.",
            chatbot=gr.Chatbot(height=500, type="messages"),
            textbox=gr.Textbox(
                placeholder="Ask a question about ROS2 navigation...",
                container=False
            ),
            type="messages"
        )

        dropdown = gr.Dropdown(
            choices=[
                "Tell me how can I navigate to a specific pose - include replanning aspects in your answer.",
                "Can you provide me with code for this task?"
            ],
            label="Common Questions",
            interactive=True
        )

        def handle_dropdown(value):
            return value

        dropdown.select(
            fn=handle_dropdown,
            inputs=[dropdown],
            outputs=[chatbot.textbox]
        )

    return demo


if __name__ == "__main__":
    # First run the pipeline
    # Task.init(project_name="ROS2_RAG", task_name="RAG_Pipeline")
    # PipelineDecorator.run_locally()
    # pipeline_controller()

    # Then start the Gradio interface
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
