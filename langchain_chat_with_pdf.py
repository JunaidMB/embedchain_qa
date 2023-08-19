from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

import os
from pprint import pprint
import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Select Embeddings model used for converting text chunks to embeddings
persist_directory = str(Path()/'docs')

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Add some PDF Documents
## Load a PDF Document
loader = PyPDFLoader(str(Path()/'pdf_docs'/'MeasuringClassifierPerformanceHMeasure.pdf'))
pages = loader.load()

print(len(pages))
pprint(pages[10])

## Split the document into chunks of size 1000 characters each with an overlap of 150 characters between chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)

print(len(pages))
print(len(docs))

pprint(docs[0].page_content)
pprint(docs[1].page_content)

# Create a ChromaDB local vectorstore to hold embeddings for chunks
## Before chunks are held in the vectorstore, they must be transformed to vector embeddings
db = Chroma.from_documents(documents = docs, embedding = embedding, persist_directory=persist_directory)

# Retrieval QA - With memory

## Select LLM to use
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

## Load LLM
llm = ChatOpenAI(model_name=llm_name, temperature=0.0)

## Initialise external memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
memory.input_key="question"
memory.output_key="answer"

# Build prompt to wrap our answers and give context
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Setup QA chain
qa_chain_with_memory = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=db.as_retriever(),
    memory=memory,
    chain_type="stuff",
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "Summarise the incoherency related to AUC?"
result = qa_chain_with_memory({"question": question})

pprint(result["chat_history"][-1].content)
pprint([doc.metadata for doc in result['source_documents']])

question_2 = "What is meant by relative severities of misclassifications"
result = qa_chain_with_memory({"question": question_2})

pprint(result["chat_history"][-1].content)
pprint([doc.metadata for doc in result['source_documents']])

question_3 = "What is AUC"
result = qa_chain_with_memory({"question": question_3})

pprint(result["chat_history"][-1].content)
pprint([doc.metadata for doc in result['source_documents']])

for idx, msg in enumerate(result['chat_history']):
    if idx %2 == 0:
        speaker = "Human"
    else:
        speaker = "AI"
    
    print(f'{speaker}: {msg.content} \n')