from query_rag import (
    get_config,
    get_graph,
    get_rate_limiter,
    get_llm,
    get_prompt,
    get_entity_chain,
    get_vector_index,
    get_template,
    get_search_query,
    get_question_template,
    get_query_chain
    )
from entity import Entity
from dotenv import load_dotenv
import langchain


import time

def call(question, chat_history = []):

    # langchain.debug = True

    print(chat_history)

    # chat_history = [(x['question'], x['answer']) for x in chat_history]
    

    load_dotenv()

    config = get_config()

    MODEL = "gemini-2.0-flash"

    graph = get_graph()

    rate_limiter = get_rate_limiter(config['RATE_LIMITS']['RPS'], config['RATE_LIMITS']['N_SECS'])


    llm = get_llm(MODEL, rate_limiter, config["TEMPERATURE"])

    # question = input("USER : ")

    prompt = get_prompt()

    start_time = time.time()
    entity_chain = get_entity_chain(prompt,llm, Entity)
    print("ENTITY_CHAIN: --- %s seconds ---" % (time.time() - start_time))

    # extracted_entities = invoke_entity(entity_chain, question)
    vector_index = get_vector_index()
    

    # final_data = retriever(question, vector_index, entity_chain, graph)

    _template = get_template()

    start_time = time.time()
    _search_query = get_search_query(_template, llm)
    print("SEARCH_QUERY: --- %s seconds ---" % (time.time() - start_time))

    template = get_question_template()

    chain = get_query_chain(_search_query, template, llm, vector_index, entity_chain, graph)


    
    start_time = time.time()
    answer = chain.invoke(
        {
            "question" : question,
            "chat_history" : chat_history
        }
        )
    
    print("OUTPUT: --- %s seconds ---" % (time.time() - start_time))

    return {
        "question" : question,
        "answer" : answer
        }