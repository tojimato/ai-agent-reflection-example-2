import os
from dotenv import load_dotenv

load_dotenv(".env.local")

WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_REGION = os.environ.get("WATSONX_REGION")
TAVILY_API_KEY= os.environ.get("TAVILY_API_KEY")