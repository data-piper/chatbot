from transformers import DistilBertTokenizer, DistilBertModel
from elasticsearch import Elasticsearch
import torch

# Initialize Elasticsearch client and DistilBERT model
es = Elasticsearch("http://localhost:9200")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def ingest_document(doc_id, text):
    """Embed and index a document in Elasticsearch."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    
    es.index(index="documents", id=doc_id, body={"text": text, "embedding": embeddings.tolist()})
    return f"Document {doc_id} ingested successfully."

def search(query, top_k=3):
    """Retrieve top_k relevant documents based on the query."""
    inputs = tokenizer(query, return_tensors="pt", truncation=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding[0].tolist()},
            }
        }
    }

    search_results = es.search(index="documents", body={"query": script_query})
    top_docs = [hit["_source"]["text"] for hit in search_results["hits"]["hits"][:top_k]]
    return "\n\n".join(top_docs) if top_docs else "No relevant documents found."
