import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(api_key = (os.getenv("GROQ_API_KEY") ), 
                base_url="https://api.groq.com/openai/v1")  #Create the OpenAI-compatible Groq client


model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs = {'normalize_embeddings':True} )

vectorStore = Chroma(
    collection_name='medical_data',
    embedding_function= model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space':'cosine'} )


def llm (query,context):

    system_prompt = f'''You are a medical information assistant. Answer the user's question using ONLY
    the provided source documents below.

      Rules you must follow:
      1. Base your answer ONLY on the provided sources — do not use outside knowledge
      2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
      3. Cite sources using [Source N] notation whenever you use information from them
      4. If sources contradict each other, mention the contradiction
      5. Be concise — answer in 3-5 sentences unless the question requires more detail
      6. Never speculate or make assumptions beyond what the sources say

      Sources:
      {context}'''

    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query} ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        messages=messages)

    return response.choices[0].message.content


def generate_id(docs):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]


def read_pdf_file(file_name):
    reader = PdfReader(file_name)

    docs = []
    for index, page in enumerate(reader.pages):
        doc = Document(
            page_content = page.extract_text(),metadata = {"source": file_name, "page": index+1})
        
        docs.append(doc)
    
    return docs

def main():
    docs = read_pdf_file("symptoms.pdf")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=110,
        chunk_overlap=10
    )
    chunks = splitter.split_documents(docs)

    #texts = [chunk.page_content for chunk in chunks]
    #vectors = model.embed_documents(texts)

    ids = generate_id(chunks)

    print(ids[0])

    vectorStore.add_documents(chunks, ids=ids)

    while True:
        user_input = input(
            'Ask something (type "q" to terminate):' ).strip().lower()

        if user_input == 'q':
            print('Good Luck')
            break

        
        retrieved_chunks = vectorStore.similarity_search_with_score(
            user_input, k=3
        )

        parts = []

        for i, (doc, score) in enumerate(retrieved_chunks):
            parts.append(
                f"[Source {i+1}] | page {doc.metadata.get('page', '?')} "
                f"| cosine distance {score:.3f}\n"
                f"{doc.page_content}"
            )

        context = "\n\n____\n\n".join(parts)

        print(llm(user_input, context))



      #  for i, chunk in enumerate(retrieved_chunks):
       #     print(f'___chunk {i}___\n')
       #     print(f'___score {chunk[1] : .3f}___\n')
       #     print(f"___Metadata : source : {chunk[0].metadata['source']} - page : {chunk[0].metadata['page']} ___ \n")
       #     print(chunk[0].page_content[0:300])

        
    #all_docs = vectorStore.get()

    #print(len(all_docs['documents']))
    #print(all_docs['documents'][0])


if __name__=='__main__':
    main()



# Project Process: RAG (Retrieval-Augmented Generation)

# 1. PDF Loading:
# 25 PDF Pages --> 25 Document Objects
# Each Document contains page_content and metadata (source, page)

# 2. Text Chunking:
# 25 Documents --> RecursiveCharacterTextSplitter
# chunk_size=110, chunk_overlap=10
# --> Smaller Document Chunks (e.g., 159 chunks)
# Each Chunk preserves its content and metadata

# 3. ID Generation:
# Each Chunk --> MD5 Hash --> Unique Content-Based ID
# Identical chunk contents produce identical IDs

# 4. Embedding Generation:
# Each Chunk --> HuggingFace Embedding Model (all-MiniLM-L6-v2)
# Each Chunk --> 384-Dimensional Vector
# 159 Chunks --> 159 Embedding Vectors

# 5. Vector Database:
# Chunks + Embeddings + Metadata + IDs --> Chroma Vector Database
# Vectors are stored locally in ./chroma
# Cosine Distance is used for similarity search

# 6. User Query:
# User enters a question --> Query Embedding
# Query Vector --> Chroma Similarity Search (k=3)
# Retrieve the 3 most relevant Document Chunks
# Lower Cosine Distance --> Higher Semantic Similarity

# 7. Context Preparation:
# Retrieved Chunks --> Extract Text + Source + Page Number
# Combine Retrieved Chunks --> Context
# Context + User Question --> LLM Prompt

# 8. LLM Response Generation:
# System Prompt + Retrieved Context + User Question
# --> Groq API --> GPT-OSS-20B Model
# --> Generate an Answer Based ONLY on Retrieved Sources
# --> Include Source Citations [Source N]

# 9. Final Output:
# LLM Response --> Display Answer to User
# If Context is Insufficient --> Return "Not Enough Information"

# 10. Interactive Loop:
# Ask Question --> Retrieve Chunks --> Generate Answer
# --> Display Answer --> Ask Next Question
# Type "q" --> Exit Program
