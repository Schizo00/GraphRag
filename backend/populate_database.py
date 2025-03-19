from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from entity import Entity

import re
import time
from tqdm_loggable.auto import tqdm
import json
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

def get_config():
    with open("./config.json", "r") as f:
        config = json.load(f)

    return config

def get_graph():
    graph = Neo4jGraph()

    return graph

def get_rate_limiter(rps, n_seconds):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=rps, 
        check_every_n_seconds=n_seconds
        )
    
    return rate_limiter
    
def get_llm(model, temp, rate_limiter=None):
    llm = ChatGoogleGenerativeAI(model=model, rate_limiter=rate_limiter, verbose=True, temperature=temp)

    return llm

def get_llm_transformer(llm):
    llm_transformer = LLMGraphTransformer(llm=llm)

    return llm_transformer


def replace_company_names(text):
    companies = [
    "Lolc Technologies Limited",
    "Lolc Technologies",
    "Lanka Orix Information Technology Services Limited",
    "Lolc Technology Limited",
    "Lolc Technology Services Ltd"
]
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(name) for name in companies) + r')\b', re.IGNORECASE)
    return pattern.sub(companies[0], text)

def load_data_from_docs(config):
    DATA_SOURCE = config['DATA']

    raw_documents = []

    for i in os.listdir(DATA_SOURCE):
        if i.endswith(".md"):
            loader = UnstructuredMarkdownLoader(f"{DATA_SOURCE}/{i}")
            data = loader.load()
            # print("DOCS TYPE: ", data)

            for doc in data:
                doc.page_content = replace_company_names(doc.page_content)

            raw_documents.extend(data)

            # raw_documents.append(data[0])

    return raw_documents


def split_documents(raw_documents):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)

    return documents


def convert_to_graph(documents, llm_transformer):
    graph_documents = []
    for i, doc in enumerate(tqdm(documents)):
        graph_documents.extend(llm_transformer.convert_to_graph_documents([doc]))
        # time.sleep(1)

    return graph_documents

def add_to_graph_db(graph, graph_documents):

    graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)
    

def get_vector_index():
    vector_index = Neo4jVector.from_existing_graph(
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
        )
    return vector_index


def run():

    load_dotenv()

    MODEL = "gemini-2.0-flash"

    config = get_config()
    graph = get_graph()

    graph.query("MATCH (n) DETACH DELETE n") # Deleting existing knowledge base

    print("\n\n\n\n", config['RATE_LIMITS']['RPS'], config['RATE_LIMITS']['N_SECS'], "\n\n\n\n")

    rate_limiter = get_rate_limiter(config['RATE_LIMITS']['RPS'], config['RATE_LIMITS']['N_SECS'])

    llm = get_llm(MODEL, 0, rate_limiter)

    llm_transformer = get_llm_transformer(llm)

    raw_documents = load_data_from_docs(config)

    documents = split_documents(raw_documents)

    graph_documents = convert_to_graph(documents, llm_transformer)

    add_to_graph_db(graph, graph_documents)

    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")



if __name__ == "__main__":

    run()

