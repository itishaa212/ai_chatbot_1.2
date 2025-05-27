# import os
# import asyncio
# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from fastapi.staticfiles import StaticFiles
# from scraper import main
# from apscheduler.schedulers.background import BackgroundScheduler
# from pinecone_search import pinecone_response
# # from serpapi_search import serpapi_search

# app = FastAPI()

# scheduler = BackgroundScheduler()

# scraping_done = False
# pinecone_index_ready = False

# async def start_scraping():
#     global scraping_done, pinecone_index_ready
#     print("Starting full scraping and indexing...")
#     await main() 
#     scraping_done = True
#     pinecone_index_ready = True
#     print("Scraping + Pinecone indexing finished.")

# async def scrape_background():
#     await start_scraping()

# @app.on_event("startup")
# async def startup_event():
#     print("Starting background scraping...")
#     await start_scraping() 

# @app.on_event("shutdown")
# async def shutdown_event():
#     if scheduler.running:
#         scheduler.shutdown()

#     print("Shutting down... Cleaning up Pinecone index.")
#     try:
#         config = dotenv_values(".pinecone_config")
#         api_key = config.get("PINECONE_API_KEY")
#         environment = config.get("PINECONE_ENVIRONMENT")
#         index_name = config.get("PINECONE_INDEX_NAME")

#         if api_key and environment and index_name:
#             pinecone.init(api_key=api_key, environment=environment)
#             if index_name in pinecone.list_indexes():
#                 pinecone.delete_index(index_name)
#                 print(f"Deleted Pinecone index: {index_name}")
#             else:
#                 print(f"Index '{index_name}' does not exist.")
#         else:
#             print("Missing Pinecone config variables.")
#     except Exception as e:
#         print(f"Error during index deletion: {e}")

# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/images", StaticFiles(directory="images"), name="images")
# templates = Jinja2Templates(directory="templates")

# class Query(BaseModel):
#     query: str  

# # Home route 
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("home.html", {"request": request})

# # Route to check if scraping is done
# @app.get("/scraping_status")
# async def scraping_status():
#     if scraping_done:
#         return {"message": "Scraping completed."}
#     else:
#         return {"message": "Scraping still in progress..."}
# # Chat page route
# @app.get("/chat", response_class=HTMLResponse)
# async def chat_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Multi-Agent System to process both responses
# async def multiagent_response(query: str) -> dict:
#     pinecone_result = await asyncio.to_thread(pinecone_response, query)
#     # serpapi_result = await asyncio.to_thread(serpapi_search, query)
#     return {
#         "pinecone_response": pinecone_result['response'],
#         # "serpapi_response": serpapi_result['response'],
#         # "references": serpapi_result['references']
#     }

# @app.post("/chat/")
# async def chat(request: Request):
#     data = await request.json()
#     query = data.get("query")
#     result = await multiagent_response(query)
#     return {
#         "response": {
#             "pinecone_response": result["pinecone_response"]
#             # "serpapi_response": result["serpapi_response"]
#         }
#     }


# @app.post("/pinecone_search")
# async def pinecone_search(query: Query):
#     if not pinecone_index_ready:
#         print("[WARNING] Tried to search before Pinecone index was ready.")
#         return {"response": "Please wait..."}  # Dummy fallback, could be blank too

#     print("[INFO] Pinecone index is ready. Processing search query.")
#     # response = pinecone_response(query.query)
#     response = await asyncio.to_thread(pinecone_response, query.query)
#     return {"response": response["response"]}

# # # Route to handle SerpAPI search
# # @app.post("/serpapi_search")
# # async def serpapi_search_endpoint(query: Query):
# #     result = serpapi_search(query.query)  # Use the Query model here
# #     return {"response":  result['response'], "references": result['references']}

#trial
# import os
# import asyncio
# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from apscheduler.schedulers.background import BackgroundScheduler
# from scraper import scrape_and_update  # Updated scraper logic
# from pinecone_search import pinecone_response
# import pinecone

# app = FastAPI()
# load_dotenv()

# # Background scheduler
# scheduler = BackgroundScheduler()

# # Flags
# scraping_done = False
# pinecone_index_ready = False

# # Startup logic
# @app.on_event("startup")
# async def startup_event():
#     global pinecone_index_ready
#     print("App startup: scraping and indexing...")

#     # Initial scrape and index creation
#     await scrape_and_update()

#     # Set up the periodic scraping task
#     scheduler.add_job(scrape_and_update, 'interval', minutes=15)
#     scheduler.start()

#     pinecone_index_ready = True
#     print("Startup indexing complete.")

# # Shutdown cleanup
# @app.on_event("shutdown")
# async def shutdown_event():
#     if scheduler.running:
#         scheduler.shutdown()

#     print("Shutting down... Cleaning up Pinecone index.")
#     try:
#         api_key = os.getenv("PINECONE_API_KEY")
#         environment = os.getenv("PINECONE_ENVIRONMENT")
#         index_name = os.getenv("PINECONE_INDEX_NAME")

#         if api_key and environment and index_name:
#             pinecone.init(api_key=api_key, environment=environment)
#             if index_name in pinecone.list_indexes():
#                 pinecone.delete_index(index_name)
#                 print(f"Deleted Pinecone index: {index_name}")
#             else:
#                 print(f"Index '{index_name}' does not exist.")
#         else:
#             print("Missing Pinecone config variables.")
#     except Exception as e:
#         print(f"Error during index deletion: {e}")

# # Static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.mount("/images", StaticFiles(directory="images"), name="images")
# templates = Jinja2Templates(directory="templates")

# # Pydantic schema
# class Query(BaseModel):
#     query: str

# # Routes
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("home.html", {"request": request})

# @app.get("/chat", response_class=HTMLResponse)
# async def chat_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/scraping_status")
# async def scraping_status():
#     if scraping_done:
#         return {"message": "Scraping completed."}
#     else:
#         return {"message": "Scraping still in progress..."}

# @app.post("/chat/")
# async def chat(request: Request):
#     data = await request.json()
#     query = data.get("query")
#     result = await multiagent_response(query)
#     return {"response": result}

# @app.post("/pinecone_search")
# async def pinecone_search(query: Query):
#     if not pinecone_index_ready:
#         return {"response": "Please wait, index not ready."}
#     response = await asyncio.to_thread(pinecone_response, query.query)
#     return {"response": response["response"]}

# # Helper
# async def multiagent_response(query: str):
#     pinecone_result = await asyncio.to_thread(pinecone_response, query)
#     return {
#         "pinecone_response": pinecone_result["response"]
#     }


import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from scraper import scrape_and_update
from pinecone_search import pinecone_response  # Use latest Pinecone search logic

app = FastAPI()
load_dotenv()

scheduler = BackgroundScheduler()
templates = Jinja2Templates(directory="templates")
pinecone_index_ready = False

# Run at server startup
@app.on_event("startup")
async def startup_event():
    global pinecone_index_ready
    print("[Startup] Running first web scrape and index creation...")

    await scrape_and_update()  # Builds new index and saves name to .pinecone_config
    pinecone_index_ready = True

  
 # Schedule auto-update every 2 hours
    scheduler.add_job(lambda: asyncio.run(scrape_and_update()), 'interval', hours=2)
    scheduler.start()


    print("[Startup] Index is ready. Scheduled future updates every 2 hour")

# Stop background scheduler on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
        print("[Shutdown] Scheduler stopped.")

# Static file routes
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Pydantic model for input
class Query(BaseModel):
    query: str

# HTML routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Main API route for Pinecone search
@app.post("/pinecone_search")
async def pinecone_search(query: Query):
    if not pinecone_index_ready:
        return {"response": "Please wait, index is still being initialized."}
    
    result = await asyncio.to_thread(pinecone_response, query.query)
    return {"response": result["response"]}

# Route to handle SerpAPI search (if needed)
# @app.post("/serpapi_search")
# async def serpapi_search_endpoint(query: Query):
#     result = serpapi_search(query.query)
#     return {"response":  result['response'], "references": result['references']}
