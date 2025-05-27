""" openai using response generated existed pinecone (after scrapping and embedding way to store)chatbot-976e9407"""
#import os
# import openai
# from pinecone import Pinecone, ServerlessSpec
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI
# from typing import Dict, Any
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize  api key 
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
# INDEX_NAME = "chatbot-976e9407"
# EMBEDDING_MODEL = "text-embedding-ada-002"
# EMBEDDING_DIM = 1536

# if INDEX_NAME not in pc.list_indexes().names():

#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=EMBEDDING_DIM,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# else:
#     print(f" Pinecone index '{INDEX_NAME}' found.")

# class QueryState:
#     def __init__(self, query: str, context: str = "", response: str = None):
#         self.query = query
#         self.context = context
#         self.response = response

# # Pinecone Search 
# def pinecone_search(user_query: str) -> QueryState:
#     index = pc.Index(INDEX_NAME)

#     print(f" Embedding user query: '{user_query}'")
#     embedding_response = openai.Embedding.create(
#         input=user_query,
#         model=EMBEDDING_MODEL,
#         api_key=OPENAI_API_KEY
#     )
#     query_embedding = embedding_response['data'][0]['embedding']


#     query_results = index.query(
#         namespace="default",  
#         vector=query_embedding,
#         top_k=4,
#         include_metadata=True
#     )

#     matches = query_results.get("matches", [])
#     if not matches:
#         print("No matches found in Pinecone.")

#     context = "\n".join(match["metadata"]["text"] for match in matches)

#     print(f"\n{context}\n")

#     return QueryState(query=user_query, context=context)

# #  LLM Response 
# def pinecone_response(user_query: str) -> Dict[str, Any]:
#     state = pinecone_search(user_query)

#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system",
#       """You are OpenEyes Assistant, a professional, knowledgeable, and helpful assistant for OpenEyes Software Solutions.,
# Always use only the provided company context to answer user questions.
# Understand and interpret short, vague, misspelled, or complex queries accurately.

# Response Logic:
# - Identify the type of user query: short, long, list-based, vague, or context-specific (e.g. article/news).
# - Decide the response structure based on the query:
#     • Short/brief → Provide a concise paragraph under 100 words.
#     • Request for 'more details' → Write 1–3 paragraphs (up to 900 words) using only the provided context.
#     • Feature/list-style → Use clear, well-spaced bullet points after a short intro paragraph.
#     • Long/complex → Begin with a short paragraph that clarifies the user’s intent, followed by 1–2 paragraphs of explanation, then supporting bullet points if needed.
#     • Article/news-related → Respond strictly based on the context; do not assume or infer outside information.


# Formatting Rules:
# - Do not use labels like "Based on the context provided" “Summary”, “Key Points”, “Conclusion”,, or numbered sections in the response.
# - Use paragraphs and bullet points for clarity, but let them appear naturally in the flow of a human conversation.
# - Never mention Pinecone, external tools, AI, or where the context came from.
# - Do not invent or assume anything not present in the context.
# - Maintain a friendly, helpful, and on-topic tone at all times.

# Error Handling:
# - For technical issues, respond with: "Oops! There was a technical issue. Please try again shortly."
# - Avoid technical language or exposing backend systems.
# Always follow this format and logic silently, without mentioning it to the user."),
#     """),
#         ("user",
#          "User Question: {query}\n\n"
#          "Company Context:\n{context}")
#     ])

#     formatted_messages = prompt_template.format_messages(
#         query=state.query,
#         context=state.context
#     )

  
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY)
#     full_answer = llm(formatted_messages).content

#     state.response = full_answer
#     print(f"\n[FINAL RESPONSE]\n{state.response}\n")

#     return {"response": state.response}

"""groq api,hugging face api  using existed pinecone based response(after scrapping and embedding way to store)"""
# import os
# import requests
# from typing import Dict, List
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain.embeddings import HuggingFaceEmbeddings

# load_dotenv()


# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# INDEX_NAME = "chatbot-4b34b99f"
# EMBEDDING_DIM = 768    

# pc = Pinecone(api_key=PINECONE_API_KEY)


# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=EMBEDDING_DIM,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# else:
#     print(f"Pinecone index '{INDEX_NAME}' found.")


# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )


# def embed_query(query: str) -> List[float]:
#     try:
#         return embedding_model.embed_query(query)
#     except Exception as e:
#         print(f"[EMBEDDING ERROR]: {e}")
#         return [0.0] * EMBEDDING_DIM  

# #  Pinecone search
# def pinecone_search(query_vector: List[float]) -> str:
#     try:
#         index = pc.Index(INDEX_NAME)
#         result = index.query(
#             namespace="default",
#             vector=query_vector,
#             top_k=4,
#             include_metadata=True
#         )
#         matches = result.get("matches", [])
#         if not matches:
#             return ""
#         context = "\n".join(match["metadata"]["text"] for match in matches)
#         print(f"\n[CONTEXT FOUND]\n{context}\n")
#         return context
#     except Exception as e:
#         print(f"[PINECONE ERROR]: {e}")
#         return ""

# #  System Prompt
# system_prompt = """You are OpenEyes Assistant, a helpful, knowledgeable assistant for OpenEyes Software Solutions.

# Always give direct, helpful answers based on the provided context. Never include information from outside the context. Never mention AI, Pinecone, or how the response was generated.

# Respond format:
# - If user asks for 'details' or 'more details' → give a comprehensive answer (200–300 words), strictly using only the context.
# - Short, meaningful query → give a clear summary using the context, then say: "Let me know if you want more details."
# - List-type query (e.g. features, steps) → start with a short intro, then use bullet points.
# - Long or complex query → identify intent, summarize clearly, and expand up to 200 words if needed.
# - Random, unclear, or gibberish query → reply kindly, suggest asking about services, features, pricing, or support.
# - No context match → say: "Sorry, I couldn't find that information right now. You can ask about our company."

# Rules:
# - Keep responses concise, simple, and user-friendly.
# - Use natural paragraphs and bullet points only when helpful.
# - Never use bold text, labels like "Summary" or "Conclusion", or mention where context came from.
# - Always stay relevant and on-topic.
# - For technical issues, reply: "Oops! There was a technical issue. Please try again shortly."
# """

# # Groq response
# def groq_response(query: str, context: str) -> str:
#     api_url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Query: {query}\n\nRelevant Context:\n{context}"}
#     ]

#     payload = {
#         "model": "llama3-70b-8192",
#         "messages": messages,
#         "temperature": 0.1
#     }

#     try:
#         response = requests.post(api_url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         return "Oops! There was a technical issue. Please try again shortly."

# #  Complete chatbot pipeline
# def pinecone_response(query: str) -> Dict[str, str]:
#     query_vector = embed_query(query)
#     context = pinecone_search(query_vector)

#     if not context.strip():
#         return {
#             "response": "Sorry, I couldn't find that information right now. You can ask about our company."
#         }

#     answer = groq_response(query, context)
#     print(f"\n[FINAL RESPONSE]\n{answer}\n")
#     return {"response": answer}

"""webscrapping way hardcoded link https://theopeneyes.com/"""
# import os 
# import asyncio
# import httpx
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin, urldefrag
# from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import hashlib
# import time
# import uuid
# import nest_asyncio
# from pinecone import Pinecone, ServerlessSpec
# import requests
# from typing import Dict, List
# from dotenv import load_dotenv
# from dotenv import dotenv_values
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# print(PINECONE_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Pinecone search for user query
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)


# # Embedding user query
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )

# def embed_query(query: str) -> List[float]:
#     try:
#         return embedding_model.embed_query(query)
#     except Exception as e:
#         print(f"[EMBEDDING ERROR]: {e}")
#         return [0.0] * 768  # Default to zero vector in case of error
# def get_index():
#     config = dotenv_values(".pinecone_config")
#     index_name = config.get("PINECONE_INDEX_NAME")
#     if not index_name:
#         raise RuntimeError("PINECONE_INDEX_NAME not found. Run the scraper first.")
#     return pc.Index(index_name)

# # Queries Pinecone for matches
# def pinecone_search(query_vector: List[float]) -> str:
#     try:
#         index = get_index()
#         result = index.query(
#             namespace="default",
#             vector=query_vector,
#             top_k=4,
#             include_metadata=True
#         )
#         matches = result.get("matches", [])
#         if not matches:
#             return ""
#         context = "\n".join(match["metadata"]["text"] for match in matches)
#         print(f"\n[CONTEXT FOUND]\n{context}\n")
#         return context
#     except Exception as e:
#         print(f"[PINECONE ERROR]: {e}")
#         return ""
# # System Prompt
# system_prompt = """You are OpenEyes Assistant, a helpful, knowledgeable assistant for OpenEyes Software Solutions.

# Always give direct, helpful answers based on the provided context. Never include information from outside the context. Never mention AI, Pinecone, or how the response was generated.

# Respond format:
# - If user asks for 'details' or 'more details' → give a comprehensive answer (200–300 words), strictly using only the context.
# - Short, meaningful query → give a clear summary using the context, then say: "Let me know if you want more details."
# - List-type query (e.g. features, steps) → start with a short intro, then use bullet points.
# - Long or complex query → identify intent, summarize clearly, and expand up to 200 words if needed.
# - Random, unclear, or gibberish query → reply kindly, suggest asking about services, features, pricing, or support.
# - No context match → say: "Sorry, I couldn't find that information right now. You can ask about our company."
# """

# # Groq response
# def groq_response(query: str, context: str) -> str:
#     api_url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Query: {query}\n\nRelevant Context:\n{context}"}
#     ]

#     payload = {
#         "model": "llama3-70b-8192",
#         "messages": messages,
#         "temperature": 0.1
#     }

#     try:
#         response = requests.post(api_url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         return "Oops! There was a technical issue. Please try again shortly."

# # Complete chatbot pipeline
# def pinecone_response(query: str) -> Dict[str, str]:
#     query_vector = embed_query(query)
#     context = pinecone_search(query_vector)

#     if not context.strip():
#         return {
#             "response": "Sorry, I couldn't find that information right now."
#         }

#     answer = groq_response(query, context)
#     print(f"\n[FINAL RESPONSE]\n{answer}\n")
#     return {"response": answer}
#testing 
import os
import requests
from typing import List, Dict
from dotenv import dotenv_values
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load API keys
config = dotenv_values(".env")
PINECONE_API_KEY = config.get("PINECONE_API_KEY")
GROQ_API_KEY = config.get("GROQ_API_KEY")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

def embed_query(query: str) -> List[float]:
    try:
        return embedding_model.embed_query(query)
    except Exception as e:
        print(f"[EMBED ERROR] {e}")
        return [0.0] * 768

def get_index():
    index_name = dotenv_values(".pinecone_config").get("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("No Pinecone index configured.")
    return pc.Index(index_name)

def pinecone_search(query_vector: List[float]) -> str:
    index = get_index()
    result = index.query(namespace="default", vector=query_vector, top_k=4, include_metadata=True)
    matches = result.get("matches", [])
    if not matches:
        return ""
    return "\n".join(match["metadata"]["text"] for match in matches)

def groq_response(query: str, context: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are OpenEyes Assistant..."},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Context:\n{context}"}
        ],
        "temperature": 0.1
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "Sorry, there was a technical issue. Try again later."

def pinecone_response(query: str) -> Dict[str, str]:
    query_vector = embed_query(query)
    context = pinecone_search(query_vector)
    if not context.strip():
        return {"response": "Sorry, I couldn't find that information right now."}
    return {"response": groq_response(query, context)}




# import os
# import time
# import json
# import hashlib
# import requests
# from typing import Dict, List
# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pinecone import Pinecone, ServerlessSpec

# # Load environment variables
# load_dotenv()

# # Initialize Pinecone and Embedding model
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )

# # Define the filename for previously seen hashes
# HASHES_FILE = "seen_hashes.json"

# # Function to save the seen hashes (to avoid duplicate processing)
# def save_seen_hashes(seen_hashes):
#     with open(HASHES_FILE, "w") as f:
#         json.dump(list(seen_hashes), f)

# # Function to load the seen hashes from file
# def load_seen_hashes():
#     if os.path.exists(HASHES_FILE):
#         with open(HASHES_FILE, "r") as f:
#             return set(json.load(f))
#     return set()

# # Embed user query
# def embed_query(query: str) -> List[float]:
#     try:
#         return embedding_model.embed_query(query)
#     except Exception as e:
#         print(f"[EMBEDDING ERROR]: {e}")
#         return [0.0] * 768  # Default to zero vector in case of error

# # Get the Pinecone index
# def get_index():
#     index_name = os.getenv("PINECONE_INDEX_NAME")
#     if not index_name:
#         raise RuntimeError("PINECONE_INDEX_NAME not found. Run the scraper first.")
#     return pc.Index(index_name)

# # Check and create Pinecone index if it doesn't exist
# def check_or_create_index(index_name: str):
#     try:
#         existing_indexes = pc.list_indexes()
#         if index_name not in existing_indexes:
#             print(f"Creating index: {index_name}")
#             pc.create_index(
#                 name=index_name,
#                 dimension=768,  # Dimension of the embedding vector
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud='aws', region='us-east-1')
#             )
#             while not pc.describe_index(index_name).status['ready']:
#                 time.sleep(1)
#             print(f"Index {index_name} created and ready.")
#         else:
#             print(f"Index {index_name} already exists.")
#     except Exception as e:
#         print(f"Error checking/creating Pinecone index: {e}")
#         raise e

# # Function to split text into chunks and prepare them for Pinecone
# def prepare_document_chunks(docs):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
#     return text_splitter.split_documents(docs)

# # Upsert documents to Pinecone (only if new or updated)
# def upsert_documents_to_pinecone(docs, index_name: str):
#     # Load the previously seen hashes
#     seen_hashes = load_seen_hashes()

#     chunks = prepare_document_chunks(docs)
#     records_to_upsert = []
    
#     for doc in chunks:
#         # Generate a hash of the document content to track changes
#         doc_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()

#         # If this document is already seen, skip it
#         if doc_hash in seen_hashes:
#             continue

#         # Add the hash to the list of seen hashes
#         seen_hashes.add(doc_hash)
        
#         # Create embedding for the document
#         embedding = embedding_model.embed_documents([doc.page_content])[0]

#         # Prepare the record to upsert
#         record = {
#             "id": f"{uuid.uuid4()}",
#             "values": embedding,
#             "metadata": {"text": doc.page_content}
#         }
#         records_to_upsert.append(record)

#     # Save the updated hashes back to file
#     save_seen_hashes(seen_hashes)

#     # Upsert records to Pinecone
#     if records_to_upsert:
#         index = pc.Index(index_name)
#         batch_size = 100
#         for i in range(0, len(records_to_upsert), batch_size):
#             batch = records_to_upsert[i:i + batch_size]
#             index.upsert(vectors=batch, namespace="default")
#         print(f"Upserted {len(records_to_upsert)} new/updated documents to Pinecone.")
#     else:
#         print("No new or updated documents to upsert.")

# # Queries Pinecone for matches based on the embedded query vector
# def pinecone_search(query_vector: List[float]) -> str:
#     try:
#         index = get_index()
#         result = index.query(
#             namespace="default",
#             vector=query_vector,
#             top_k=4,
#             include_metadata=True
#         )
#         matches = result.get("matches", [])
#         if not matches:
#             return ""
#         context = "\n".join(match["metadata"]["text"] for match in matches)
#         print(f"\n[CONTEXT FOUND]\n{context}\n")
#         return context
#     except Exception as e:
#         print(f"[PINECONE ERROR]: {e}")
#         return ""

# # Groq response generator
# def groq_response(query: str, context: str) -> str:
#     api_url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
#         "Content-Type": "application/json"
#     }

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Query: {query}\n\nRelevant Context:\n{context}"}
#     ]

#     payload = {
#         "model": "llama3-70b-8192",
#         "messages": messages,
#         "temperature": 0.1
#     }

#     try:
#         response = requests.post(api_url, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         return "Oops! There was a technical issue. Please try again shortly."

# # System prompt for generating detailed responses
# system_prompt = """You are OpenEyes Assistant, a helpful, knowledgeable assistant for OpenEyes Software Solutions.

# Always give direct, helpful answers based on the provided context. Never include information from outside the context. Never mention AI, Pinecone, or how the response was generated.

# Respond format:
# - If user asks for 'details' or 'more details' → give a comprehensive answer (200–300 words), strictly using only the context.
# - Short, meaningful query → give a clear summary using the context, then say: "Let me know if you want more details."
# - List-type query (e.g. features, steps) → start with a short intro, then use bullet points.
# - Long or complex query → identify intent, summarize clearly, and expand up to 200 words if needed.
# - Random, unclear, or gibberish query → reply kindly, suggest asking about services, features, pricing, or support.
# - No context match → say: "Sorry, I couldn't find that information right now. You can ask about our company."
# """

# # Complete pipeline for answering queries with context from Pinecone and Groq
# def pinecone_response(query: str) -> Dict[str, str]:
#     query_vector = embed_query(query)
#     context = pinecone_search(query_vector)

#     if not context.strip():
#         return {
#             "response": "Sorry, I couldn't find that information right now."
#         }

#     answer = groq_response(query, context)
#     print(f"\n[FINAL RESPONSE]\n{answer}\n")
#     return {"response": answer}


# def main():
#     while True:  # This is the while loop
#         # Ask the user for a query
#         query = input("Ask a question (or type 'exit' to quit): ")

#         # If the user types 'exit', break the loop
#         if query.lower() == 'exit':
#             print("Exiting chatbot.")
#             break  # Correctly placed inside the loop

#         # Generate response using the pinecone_response function
#         response = pinecone_response(query)

#         # Output the chatbot's response
#         print(f"Chatbot Response: {response['response']}")

# if __name__ == "__main__":
#     main()

