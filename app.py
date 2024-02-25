from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
import chainlit as cl

import streamlit as st
import base64

import yaml

DB_FAISS_PATH = 'faiss_vectorstore/'


custom_prompt_template = """You are a musuem guide. Use the following template to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

"""

OPENAI_CONFIG_FILE = 'api_key.yaml'

with open(OPENAI_CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

apikey = config['openai']['access_key']


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 5}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the LLM model
def load_llm():
    # Load the locally downloaded model here
    llm = OpenAI(openai_api_key=apikey)
    return llm

# QA Model Function
def qa_bot():
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
     #                                  model_kwargs={'device': 'cpu'})
    embeddings = OpenAIEmbeddings(openai_api_key = apikey)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Formulating response
def get_response(qa_result, query):
    #qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    


def main():
    st.title("Welcome to Chhatrapati shivaji maharaj vastu sangrahalaya. What story would you like to know?")
    qa = qa_bot()
    
    # Input box for user query
    user_query = st.text_input("You:", "")

    # Button to submit query
    if st.button("Generate Story"):
        response = get_response(qa, user_query)
        st.text_area("Stories:", value=response, height=200)

if __name__ == "__main__":
    add_bg_from_local('chhatrapati shivaji maharaj vastu sangrahalaya.jpeg')  # Add background image
    main()

