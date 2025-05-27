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
# from dotenv import load_dotenv
# from dotenv import dotenv_values

# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=PINECONE_API_KEY)

# CONCURRENCY = 10
# CRAWL_DELAY = 0.1
# MAX_DEPTH = 50
# BASE_URL = "https://theopeneyes.com/"

# visited_urls = set()
# seen_hashes = set()
# docs = []

# def normalize_url(url):
#     url, _ = urldefrag(url)
#     return url.rstrip('/')

# def hash_content(text):
#     return hashlib.sha256(text.encode("utf-8")).hexdigest()

# async def fetch(client, url):
#     try:
#         response = await client.get(url, timeout=10.0)
#         if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
#             return response.text
#     except Exception as e:
#         print(f" Failed: {url} → {e}")
#     return None

# async def crawl_worker(client, queue, base_domain, semaphore):
#     while True:
#         try:
#             url, depth = await queue.get()
#         except asyncio.CancelledError:
#             break

#         norm_url = normalize_url(url)

#         if norm_url in visited_urls or depth > MAX_DEPTH:
#             queue.task_done()
#             continue

#         visited_urls.add(norm_url)
#         print(f" Scraping [Depth {depth}]: {norm_url}")

#         async with semaphore:
#             html = await fetch(client, norm_url)

#         if html:
#             soup = BeautifulSoup(html, 'html.parser')
#             text = soup.get_text(separator=' ', strip=True)
#             cleaned_text = text.replace('\n', '').replace('\r', '').replace('\t', '')

#             content_hash = hash_content(cleaned_text)
#             if content_hash not in seen_hashes and cleaned_text.strip():
#                 seen_hashes.add(content_hash)
#                 docs.append(Document(page_content=cleaned_text, metadata={"url": norm_url}))

#             for tag in soup.find_all('a', href=True):
#                 href = tag['href']
#                 if href.startswith(('mailto:', 'tel:', 'javascript:')): 
#                     continue
#                 full_url = urljoin(norm_url, href)
#                 parsed = urlparse(full_url)
#                 if parsed.netloc.endswith(base_domain):
#                     cleaned = normalize_url(full_url)
#                     if cleaned not in visited_urls:
#                         queue.put_nowait((cleaned, depth + 1))

#         await asyncio.sleep(CRAWL_DELAY)
#         queue.task_done()

# async def scrape_to_docs(base_url):
#     domain = urlparse(base_url).netloc
#     queue = asyncio.Queue()
#     await queue.put((base_url, 0))
#     semaphore = asyncio.Semaphore(CONCURRENCY)

#     async with httpx.AsyncClient(follow_redirects=True) as client:
#         tasks = [asyncio.create_task(crawl_worker(client, queue, domain, semaphore)) for _ in range(CONCURRENCY)]
#         await queue.join()
#         for task in tasks:
#             task.cancel()

#     return docs

# def batch(iterable, batch_size=100):
#     for i in range(0, len(iterable), batch_size):
#         yield iterable[i:i + batch_size]

# async def create_index_and_upsert(docs):
#     print(f"\n Total unique documents scraped: {len(docs)}")

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         length_function=len,
#     )
#     chunks = text_splitter.split_documents(docs)
#     print(f" Total chunks: {len(chunks)}")

#     index_name = f"chatbot-{uuid.uuid4().hex[:8]}"
#     print(f" Creating Pinecone index: {index_name}")

#     # Create Pinecone index
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud='aws', region='us-east-1')
#     )
#     while not pc.describe_index(index_name).status['ready']:
#         time.sleep(1)
#     print(f" Index ready: {index_name}")

#     # Generate embeddings
#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": False}
#     )

#     chunks_with_ids = [{"id": f"vec{i+1}", "text": doc.page_content} for i, doc in enumerate(chunks)]
#     texts = [c['text'] for c in chunks_with_ids]

#     print("Generating embeddings...")
#     embeddings = embedding_model.embed_documents(texts)

#     # Upsert vectors to Pinecone
#     index = pc.Index(index_name)
#     records = [
#         {"id": chunk["id"], "values": embedding, "metadata": {"text": chunk["text"]}}
#         for chunk, embedding in zip(chunks_with_ids, embeddings)
#     ]

#     # Upsert records in batches
#     for batch_records in batch(records):
#         index.upsert(vectors=batch_records, namespace="default")

#     print("Vectors upserted to Pinecone successfully.")
#     print(f"INDEX_NAME={index_name}")

   
#     with open(".pinecone_config", "w") as f:
#         f.write(f"PINECONE_INDEX_NAME={index_name}\n")

# async def main():

#     try:
#         await scrape_to_docs(BASE_URL)
#     except RuntimeError:
#         nest_asyncio.apply()
#         await scrape_to_docs(BASE_URL)


#     await create_index_and_upsert(docs)
# # import os
# # import asyncio
# # import httpx
# # from bs4 import BeautifulSoup
# # from urllib.parse import urlparse, urljoin, urldefrag
# # from langchain.schema import Document
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import hashlib
# # import time
# # import uuid
# # import nest_asyncio
# # from pinecone import Pinecone, ServerlessSpec
# # from dotenv import load_dotenv
# # import feedparser
# # from datetime import datetime

# # # Load environment variables
# # load_dotenv()

# # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# # PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # # Pinecone initialization
# # pc = Pinecone(api_key=PINECONE_API_KEY)

# # # Settings
# # CONCURRENCY = 10
# # CRAWL_DELAY = 0.1
# # MAX_DEPTH = 50
# # BASE_URL = "https://theopeneyes.com/"  

# # visited_urls = set()
# # seen_hashes = set()
# # docs = []
# # rss_feeds = []

# # def normalize_url(url):
# #     """Normalize the URL by removing fragments and unnecessary slashes."""
# #     url, _ = urldefrag(url)
# #     return url.rstrip('/')

# # def hash_content(text):
# #     """Generate a SHA256 hash for the given content."""
# #     return hashlib.sha256(text.encode("utf-8")).hexdigest()

# # def parse_rss_feed_date(date_str):
# #     """Parse date from RSS feed entries."""
# #     try:
# #         return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
# #     except ValueError:
# #         return None

# # async def fetch(client, url):
# #     """Fetch HTML content from a URL."""
# #     try:
# #         response = await client.get(url, timeout=10.0)
# #         if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
# #             return response.text
# #     except Exception as e:
# #         print(f"Failed to fetch URL {url}: {e}")
# #     return None

# # def is_rss_feed(url):
# #     """Check if the URL points to an RSS feed."""
# #     return url.endswith(".xml") or "rss" in url.lower()

# # async def crawl_worker(client, queue, base_domain, semaphore):
# #     """Worker function for crawling and scraping URLs."""
# #     while True:
# #         try:
# #             url, depth = await queue.get()
# #         except asyncio.CancelledError:
# #             break

# #         norm_url = normalize_url(url)

# #         if norm_url in visited_urls or depth > MAX_DEPTH:
# #             queue.task_done()
# #             continue

# #         visited_urls.add(norm_url)
# #         print(f"Scraping [Depth {depth}]: {norm_url}")

# #         async with semaphore:
# #             html = await fetch(client, norm_url)

# #         if html:
# #             soup = BeautifulSoup(html, 'html.parser')
# #             text = soup.get_text(separator=' ', strip=True)
# #             cleaned_text = text.replace('\n', '').replace('\r', '').replace('\t', '')

# #             # Look for RSS feeds in <link> tags
# #             for tag in soup.find_all('link', rel='alternate'):
# #                 href = tag.get('href', '')
# #                 if is_rss_feed(href):
# #                     rss_feed_url = urljoin(norm_url, href)
# #                     if rss_feed_url not in rss_feeds:
# #                         rss_feeds.append(rss_feed_url)
# #                         print(f"Found RSS feed: {rss_feed_url}")

# #             content_hash = hash_content(cleaned_text)
# #             if content_hash not in seen_hashes and cleaned_text.strip():
# #                 seen_hashes.add(content_hash)
# #                 docs.append(Document(page_content=cleaned_text, metadata={"url": norm_url}))

# #             # Recursively crawl linked pages within the same domain
# #             for tag in soup.find_all('a', href=True):
# #                 href = tag['href']
# #                 if href.startswith(('mailto:', 'tel:', 'javascript:')):
# #                     continue
# #                 full_url = urljoin(norm_url, href)
# #                 parsed = urlparse(full_url)
# #                 if parsed.netloc.endswith(base_domain):  # Same domain
# #                     cleaned = normalize_url(full_url)
# #                     if cleaned not in visited_urls and depth + 1 <= MAX_DEPTH:
# #                         queue.put_nowait((cleaned, depth + 1))

# #                     await asyncio.sleep(CRAWL_DELAY)
# #                     queue.task_done()

# # async def scrape_to_docs(base_url):
# #     """Main function to scrape a website and return documents."""
# #     domain = urlparse(base_url).netloc
# #     queue = asyncio.Queue()
# #     await queue.put((base_url, 0))  # Starting URL with depth 0
# #     semaphore = asyncio.Semaphore(CONCURRENCY)

# #     async with httpx.AsyncClient(follow_redirects=True) as client:
# #         tasks = [asyncio.create_task(crawl_worker(client, queue, domain, semaphore)) for _ in range(CONCURRENCY)]
# #         await queue.join()  # Wait for all tasks to finish
# #         for task in tasks:
# #             task.cancel()  # Cancel the worker tasks after completion

# #     return docs

# # def fetch_rss_feed_content(rss_feed_url):
# #     """Fetch and parse content from an RSS feed."""
# #     feed = feedparser.parse(rss_feed_url)
# #     rss_content = []
# #     latest_date = None

# #     for entry in feed.entries:
# #         entry_date = parse_rss_feed_date(entry.published)
# #         if entry_date:
# #             if latest_date is None or entry_date > latest_date:
# #                 latest_date = entry_date
# #                 rss_content = [{
# #                     "title": entry.title,
# #                     "link": entry.link,
# #                     "summary": entry.summary,
# #                     "published": entry.published,
# #                 }]
# #             elif entry_date == latest_date:
# #                 rss_content.append({
# #                     "title": entry.title,
# #                     "link": entry.link,
# #                     "summary": entry.summary,
# #                     "published": entry.published,
# #                 })
# #     return rss_content

# # def batch(iterable, batch_size=100):
# #     """Yield successive batches from iterable."""
# #     for i in range(0, len(iterable), batch_size):
# #         yield iterable[i:i + batch_size]

# # async def create_index_and_upsert(docs):
# #     """Create a Pinecone index and upsert documents to it."""
# #     print(f"\nTotal unique documents scraped: {len(docs)}")
    
# #     # Split documents into smaller chunks
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000,
# #         chunk_overlap=100,
# #         length_function=len,
# #     )
# #     chunks = text_splitter.split_documents(docs)
# #     print(f"Total chunks: {len(chunks)}")

# #     # Generate a unique index name
# #     index_name = f"chatbot-{uuid.uuid4().hex[:8]}"
# #     print(f"Creating Pinecone index: {index_name}")
    
# #     # Create the Pinecone index
# #     pc.create_index(
# #         name=index_name,
# #         dimension=768,
# #         metric="cosine",
# #         spec=ServerlessSpec(cloud='aws', region='us-east-1')
# #     )
    
# #     # Wait for the index to be ready
# #     while not pc.describe_index(index_name).status['ready']:
# #         time.sleep(1)
# #     print(f"Index ready: {index_name}")

# #     # Embedding model for document vectors
# #     embedding_model = HuggingFaceEmbeddings(
# #         model_name="sentence-transformers/all-mpnet-base-v2",
# #         model_kwargs={"device": "cpu"},
# #         encode_kwargs={"normalize_embeddings": False}
# #     )

# #     # Prepare documents for upserting
# #     chunks_with_ids = [{"id": f"vec{i+1}", "text": doc.page_content} for i, doc in enumerate(chunks)]
# #     texts = [c['text'] for c in chunks_with_ids]

# #     # Generate embeddings for document chunks
# #     print("Generating embeddings...")
# #     embeddings = embedding_model.embed_documents(texts)

# #     # Prepare Pinecone upsert records
# #     index = pc.Index(index_name)
# #     records = [
# #         {"id": chunk["id"], "values": embedding, "metadata": {"text": chunk["text"]}}
# #         for chunk, embedding in zip(chunks_with_ids, embeddings)
# #     ]

# #     # Upsert in batches to Pinecone
# #     for batch_records in batch(records):
# #         index.upsert(vectors=batch_records, namespace="default")

# #     print("Vectors upserted to Pinecone successfully.")
# #     print(f"INDEX_NAME={index_name}")

# #     # Save Pinecone index name in a file for future reference
# #     with open(".pinecone_config", "w") as f:
# #         f.write(f"PINECONE_INDEX_NAME={index_name}\n")

# # # Final coroutine to orchestrate everything
# # async def scrape_and_update():
# #     global docs, visited_urls, seen_hashes

# #     print("Starting scrape and index update...")

# #     # Clear previous crawl state
# #     docs = []
# #     visited_urls.clear()
# #     seen_hashes.clear()

# #     # Step 1: Scrape main site
# #     new_docs = await scrape_to_docs(BASE_URL)
# #     docs.extend(new_docs)

# #     # Step 2: Fetch and append RSS content
# #     for rss_feed_url in rss_feeds:
# #         print(f"Fetching RSS feed content from {rss_feed_url}")
# #         rss_content = fetch_rss_feed_content(rss_feed_url)
# #         for entry in rss_content:
# #             content_hash = hash_content(entry['summary'])
# #             if content_hash not in seen_hashes:
# #                 seen_hashes.add(content_hash)
# #                 docs.append(Document(page_content=entry['summary'], metadata={"url": entry['link']}))

# #     # Step 3: Upsert to Pinecone
# #     await create_index_and_upsert(docs)
import os
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import uuid
import feedparser
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BASE_URL = os.getenv("SCRAPER_BASE_URL", "https://theopeneyes.com/")

pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

CONCURRENCY = 10
MAX_DEPTH = 50
CRAWL_DELAY = 0.1

visited_urls = set()
seen_hashes = set()
rss_feeds = []
latest_rss_timestamp = None

def normalize_url(url):
    url, _ = urldefrag(url)
    return url.rstrip('/')

def hash_content(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_rss_date(date_str):
    try:
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except:
        return None

async def fetch(client, url):
    try:
        response = await client.get(url, timeout=10.0)
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            return response.text
    except:
        return None

async def crawl(client, queue, domain, semaphore, docs):
    while True:
        try:
            url, depth = await queue.get()
        except asyncio.CancelledError:
            break

        norm_url = normalize_url(url)
        if norm_url in visited_urls or depth > MAX_DEPTH:
            queue.task_done()
            continue

        visited_urls.add(norm_url)
        print(f"[CRAWLING] {norm_url}")

        async with semaphore:
            html = await fetch(client, norm_url)

        if html:
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

            for tag in soup.find_all('link', rel='alternate'):
                href = tag.get('href', '')
                if "rss" in href.lower() or href.endswith('.xml'):
                    full_rss = urljoin(norm_url, href)
                    if full_rss not in rss_feeds:
                        rss_feeds.append(full_rss)
                        print(f"[FOUND RSS] {full_rss}")

            content_hash = hash_content(text)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                docs.append(Document(page_content=text, metadata={"url": norm_url}))

            for tag in soup.find_all('a', href=True):
                href = tag['href']
                if href.startswith(('mailto:', 'tel:', 'javascript:')):
                    continue
                full_url = urljoin(norm_url, href)
                parsed = urlparse(full_url)
                if parsed.netloc.endswith(domain):
                    cleaned = normalize_url(full_url)
                    if cleaned not in visited_urls:
                        queue.put_nowait((cleaned, depth + 1))

        queue.task_done()
        await asyncio.sleep(CRAWL_DELAY)

async def scrape_to_docs():
    docs = []
    domain = urlparse(BASE_URL).netloc
    queue = asyncio.Queue()
    await queue.put((BASE_URL, 0))
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [asyncio.create_task(crawl(client, queue, domain, semaphore, docs)) for _ in range(CONCURRENCY)]
        await queue.join()
        for task in tasks:
            task.cancel()

    return docs

def get_rss_entries():
    global latest_rss_timestamp
    new_docs = []

    for feed_url in rss_feeds:
        print(f"[RSS] Checking feed: {feed_url}")
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            entry_date = parse_rss_date(entry.get("published", ""))
            if entry_date and (latest_rss_timestamp is None or entry_date > latest_rss_timestamp):
                latest_rss_timestamp = entry_date
                content = entry.get("summary", "") or entry.get("description", "")
                if content:
                    content_hash = hash_content(content)
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        new_docs.append(Document(page_content=content, metadata={"url": entry.link}))

    if new_docs:
        print(f"[RSS] New entries found at {latest_rss_timestamp}")
    else:
        print("[RSS] Not coming new content. No any changes.")

    return new_docs

def batch(iterable, size=100):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def get_existing_index_name():
    if os.path.exists(".pinecone_config"):
        with open(".pinecone_config") as f:
            for line in f:
                if line.startswith("PINECONE_INDEX_NAME="):
                    return line.split("=", 1)[1].strip()
    return None

async def create_index_and_upsert(docs: List[Document], index_name: str, create_if_missing: bool = True):
    if create_if_missing:
        if index_name not in pc.list_indexes().names():
            print(f"[PINECONE] Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"[PINECONE] Index '{index_name}' already exists.")
    else:
        print(f"[PINECONE] Using existing index: {index_name}")

    index = pc.Index(index_name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[CHUNKS] Total: {len(chunks)}")

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)

    vectors = [
        {"id": f"doc-{uuid.uuid4().hex}", "values": vec, "metadata": {"text": chunk.page_content}}
        for vec, chunk in zip(embeddings, chunks)
    ]

    for batch_vecs in batch(vectors, size=100):
        index.upsert(vectors=batch_vecs, namespace="default")

    if create_if_missing:
        with open(".pinecone_config", "w") as f:
            f.write(f"PINECONE_INDEX_NAME={index_name}")

    print(f"[DONE] Upsert complete: {len(vectors)} vectors")

async def scrape_and_update():
    print("[START] Scraping and indexing process triggered.")
    scraped_docs = await scrape_to_docs()
    rss_docs = get_rss_entries()
    all_docs = scraped_docs + rss_docs

    if not all_docs:
        print("[INFO] No new documents to index.")
        return

    existing_index = get_existing_index_name()
    if existing_index:
        await create_index_and_upsert(all_docs, index_name=existing_index, create_if_missing=False)
    else:
        index_name = f"index-{uuid.uuid4().hex[:8]}"
        await create_index_and_upsert(all_docs, index_name=index_name, create_if_missing=True)
