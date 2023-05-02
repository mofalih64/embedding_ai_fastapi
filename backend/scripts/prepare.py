# imports
import openai
import pandas as pd
import tiktoken
from decouple import config

# to run this script
openai.api_key = config('OPENAI_API_KEY')
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# change this to the path of the csv file that you want to embed
input_datapath = "data_embedded.csv"

# this script works with csv file that has 2 columns: completion and prompt
df = pd.read_csv(input_datapath)
df = df[["completion", "prompt"]]
df = df.dropna()

top_n = 7000

encoding = tiktoken.get_encoding(embedding_encoding)
df["combined"] = (
    "prompt: " + df.prompt.str.strip() + "; completion: " + df.prompt.str.strip()
)

# we can also filter out the long data here
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)


# get the embedding using openai  by transfer the combined column data (the data that we will do our querying upon) , this embedding is the core element of the searching and the ranking
df["embedding"] = df.combined.apply(lambda x: openai.Embedding.create(
    input=[x], model=embedding_model)['data'][0]['embedding'])
df.to_csv("embedded_data.csv")
