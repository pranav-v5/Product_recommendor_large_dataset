import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# os.environ["GOOGLE_API_KEY"] = "AIzaSyCZaFcdLQfJzj8ZnfXT3Dnr18qK_hkcVRM"

# 1> Choose LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)

# model = genai.GenerativeModel("gemini-1.5-flash")


# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.0-pro",
#     temperature=0.2
# )


# 2> Load Data

loader = TextLoader("data/products_data.txt", encoding="utf-8")
documents = loader.load()

# 3> Split Data

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# 4> Create Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5> Store in Vector DB

vectorstore = FAISS.from_documents(docs, embeddings)

# 6> Create Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# 7> Accept User Query

query = input("Enter your product requirement: ")

# 8> Retrieve Relevant Data

relevant_docs = retriever.invoke(query)


# 9> Inject Context into Prompt

context = "\n\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""
You are a product recommendation assistant.

Based ONLY on the following products:
{context}

Recommend the best products for the user query:
"{query}"

Explain briefly why you recommend them.
"""

# 10> Generate Response

response = llm.invoke(prompt)
print(response.content)
