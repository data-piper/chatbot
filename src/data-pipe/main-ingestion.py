from elasticsearch import Elasticsearch
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Initialize Elasticsearch client
client = Elasticsearch(
    "https://your-elasticsearch-url:443",
    api_key=("your-api-id", "your-api-key")
)

# Initialize DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# List of documents to index (replace with your actual documents)
documents = [
    {"id": "doc1", "text": "Sample document text for job interviews and preparation."},
    {"id": "doc2", "text": "Another document about AI and machine learning interview questions."}
    # Add more documents as needed
]

# Ingest documents into Elasticsearch
for doc in documents:
    # Generate embedding
    inputs = tokenizer(doc["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()  # 768-dimensional vector

    # Index document in Elasticsearch
    client.index(index="nt-gen-index-1", id=doc["id"], body={
        "text": doc["text"],
        "vector": embedding
    })

    print(f"Indexed document {doc['id']} with text and vector.")
