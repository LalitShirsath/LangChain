import os
from secret_keys import openai_api_key
from langchain.llms import OpenAI
import streamlit as sl

from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = openai_api_key

sl.title("Langchain Demonstration")
input_text = sl.text_input("Search the topic")

llm = OpenAI(temperature=0.8)

person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='description_history')


prompt1 = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

chain1 = LLMChain(llm=llm,prompt=prompt1,verbose=True,output_key='person',memory=person_memory)

prompt2 = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2 = LLMChain(llm=llm,prompt=prompt2,verbose=True, output_key='dob',memory=dob_memory)

prompt3 = PromptTemplate(
    input_variables=['dob'],
    template="tell me 5 major moments happened around {dob} in india"
)

chain3 = LLMChain(llm=llm,prompt=prompt3,verbose=True,output_key='description', memory=descr_memory)

parent_chain = SequentialChain(chains=[chain1,chain2,chain3], input_variables=['name'], output_variables=['person','dob','description'], verbose=True)

if input_text:
    sl.write(parent_chain({'name': input_text}))

    with sl.expander('Person Name'):
        sl.info(person_memory.buffer)


    with sl.expander('Major Moments'):
        sl.info(descr_memory.buffer)

    