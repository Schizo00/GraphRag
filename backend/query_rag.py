from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from entity import Entity
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_neo4j import Neo4jGraph


from typing import List, Tuple
import json
import time
import asyncio

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
    
def get_llm(model, rate_limiter=None, temperature = 0):
    llm = ChatGoogleGenerativeAI(model=model, rate_limiter=rate_limiter, verbose=True, temperature=temperature)

    return llm



def get_vector_index():
    vector_index = Neo4jVector.from_existing_graph(
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
        )
    return vector_index



def get_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    return prompt

def get_entity_chain(prompt, llm, Entity):
    entity_chain = prompt | llm.with_structured_output(Entity)

    return entity_chain

def invoke_entity(entity_chain, question):
    extracted_entities = entity_chain.invoke({"question": question}).names

    return extracted_entities

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    
    words = [el for el in remove_lucene_chars(input).split() if el]

    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"

    return full_text_query.strip()

async def structured_retriever(question: str, entity_chain, graph) -> str:
    start_time = time.time()
    result = ""
    entities = entity_chain.invoke({"question": question})
    entity_queries = [generate_full_text_query(entity) for entity in entities.names]

    if entity_queries:
        query = """
        UNWIND $queries AS query
        CALL db.index.fulltext.queryNodes('entity', query, {limit:2})
        YIELD node, score
        CALL (node, score) {
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
        }
        RETURN output LIMIT 25
        """
        response = graph.query(query, {"queries": entity_queries})
        result = "\n".join([el['output'] for el in response])

    print("OPTIMIZED_STRUCTURED_RETRIEVER: --- %s seconds ---" % (time.time() - start_time))
    return result

async def async_retriever(question: str, vector_index, entity_chain, graph):
    start_time = time.time()
    # print(f"Search query: {question}")

    structured_task = asyncio.create_task(structured_retriever(question, entity_chain, graph))
    unstructured_task = asyncio.create_task(asyncio.to_thread(lambda: [el.page_content for el in vector_index.similarity_search(question)]))

    structured_data = await structured_task
    unstructured_data = await unstructured_task
    final_data = f"""
    Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
    """
    print("OPTIMIZED_FULL_RETRIEVER: --- %s seconds ---" % (time.time() - start_time))
    return final_data

def retriever(question: str, vector_index, entity_chain, graph):
    return asyncio.run(async_retriever(question, vector_index, entity_chain, graph))


def get_template():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    return PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def get_search_query(CONDENSE_QUESTION_PROMPT, llm):
    start_time = time.time()
    _search_query = RunnableBranch(
    # If input includes chat_history, include it in the follow up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
    )
    print("SEARCH_QUERY: --- %s seconds ---" % (time.time() - start_time))

    return _search_query

def get_question_template():
    template = """You are a helpful front desk assistant for LOLC Technologies. Your responses should convey that you are part of the company. Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Break down the question into multiple steps and answer. Use natural language and be concise. Do not include thinking in your answer.
    Answer:"""

    return ChatPromptTemplate.from_template(template)

def get_query_chain(_search_query, template, llm, vector_index, entity_chain, graph):
    return (
        {
            "context": _search_query | RunnableLambda(lambda q: retriever(q, vector_index, entity_chain, graph)),
            "question": RunnablePassthrough()
        }
        | template
        | llm
        | StrOutputParser()
    )


def run():

    from dotenv import load_dotenv

    load_dotenv()
    
    config = get_config()

    MODEL = "gemini-2.0-flash"

    graph = get_graph()

    rate_limiter = get_rate_limiter(config['RATE_LIMITS']['RPS'], config['RATE_LIMITS']['N_SECS'])

    llm = get_llm(MODEL, rate_limiter, config['TEMPERATURE'])

    chat_history = []

    while True:

        

        question = input("USER : ")

        prompt = get_prompt()

        entity_chain = get_entity_chain(prompt,llm, Entity)
        # extracted_entities = invoke_entity(entity_chain, question)
        vector_index = get_vector_index()

        # final_data = retriever(question, vector_index, entity_chain, graph)

        _template = get_template()

        _search_query = get_search_query(_template, llm)

        template = get_question_template()

        chain = get_query_chain(_search_query, template, llm, vector_index, entity_chain, graph)
        start_time = time.time()
        answer = chain.invoke(
            {
                "question" : question,
                "chat_history" : chat_history
            }
            )

        

        # print("\n\n\n\n", final_data)
        print(f"\n\n\n\nBOT : {answer}")

        chat_history.append((question, answer))

        print(f"FULL ANSWER_TIME: {time.time() - start_time}")


if __name__ == "__main__":
    run()



