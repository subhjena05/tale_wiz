import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from PyPDF2 import PdfReader

#TEXT READING
from langchain.document_loaders import PyPDFLoader
reader = PdfReader('SocScienceDocUpdated.pdf')

from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

#TEXT SPLITTING
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 350,
    chunk_overlap = 35
)
splits = text_splitter.split_text(raw_text)
print(f"THERE ARE {len(splits)} TEXT CHUNKS.")

#EMBEDDINGS
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#VECTORSTORES
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(splits, embedding=embeddings)

#MODEL INITIALIZATION
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

#MEMORY
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#RETRIEVAL
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)

#QUERY
query = input("What story would you like to hear about? ")
result = conversation_chain({"question": query})
answer = result["answer"]

print(answer)