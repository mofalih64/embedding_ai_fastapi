from fastapi import APIRouter
import openai
import pandas as pd
import ast
from scipy import spatial  # for calculating vector similarities for search

import tiktoken  # for converting embeddings saved as strings back to arrays
from decouple import config
from embedding.schemas import question, returnResponse
from embedding.service import ask
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = config('OPENAI_API_KEY')
# from models.user import User

# ___________________________  change this to the path of the csv file that you embedded ___________________________
df = pd.read_csv(
    'embedded_data.csv', encoding='utf-8')
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)


Embedding_router = APIRouter()


@Embedding_router.post("/ask/")
async def register(ask_in: question):
    answer = ask(ask_in.question, df=df, model=GPT_MODEL,
                 token_budget=4096 - 500, print_message=False)
    answer = answer.replace('\n', '<br>')
    return {"answer": answer}
