# import requests
# import os
# from dotenv import load_dotenv
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain

# load_dotenv()

# # Set the SEaRPAPI_KEY environment variable or hardcode the key directly (not recommended for production).
# SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Make sure your SERPAPI_KEY is set in your environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Assuming you're using OpenAI API for LLM

# # Initialize the OpenAI model via langchain
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# # Define the ChatPromptTemplate
# llmprompt_template2 = ChatPromptTemplate([
#     ("system", "You are a helpful assistant short description external knowledge."),
#     ("user", "Tell me about {external_knowledge}, or {real_time_questions}. Use the provided information to generate a helpful, accurate, and concise answer.")
# ])

# # Define the LLMChain
# llm_chain = LLMChain(prompt=llmprompt_template2, llm=llm)

# def serpapi_search(query: str) -> dict:
#     try:
#         # Request from SerpAPI
#         response = requests.get(f"https://serpapi.com/search?q={query}&api_key={SERPAPI_KEY}&engine=google")
#         data = response.json()

#         # Extract organic results from the SerpAPI response
#         organic_results = data.get("organic_results", [])
#         wikipedia_result = None

#         for result in organic_results:
#             if 'wikipedia.org' in result.get("link", ""):
#                 wikipedia_result = result
#                 break

#         result_text = ""
#         references = []

#         # If a Wikipedia result is found
#         if wikipedia_result:
#             wiki_url = wikipedia_result.get("link")
#             result_text += fetch_wikipedia_content(wiki_url)
#             references.append({"title": "Wikipedia", "link": wiki_url})
#         else:
#             for result_item in organic_results[:4]:
#                 snippet = result_item.get("snippet", "")
#                 result_text += f"\n{snippet}\n\n"
#                 references.append({"title": result_item.get("title"), "link": result_item.get("link")})

#         # Format the input for LLM
#         prompt_input = {
#             "external_knowledge": result_text,
#             "real_time_questions": query
#         }

#         # Generate the final response using LLMChain
#         final_answer = llm_chain.run(prompt_input)
#         # print(references)

#         return {"response": {"text": final_answer}, "references": references}

#     except Exception as e:
#         return {"response": {"text": f"**Error during search**: {str(e)}"}, "references": []}

# def fetch_wikipedia_content(wiki_url: str) -> str:
#     title = wiki_url.split("/")[-1]
#     wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles={title}"
#     response = requests.get(wikipedia_api_url)
#     data = response.json()
#     pages = data.get("query", {}).get("pages", {})
#     page = next(iter(pages.values()))
#     return page.get("extract", "")
