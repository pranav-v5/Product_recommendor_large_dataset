import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# os.environ["GOOGLE_API_KEY"] = "AIzaSyCZaFcdLQfJzj8ZnfXT3Dnr18qK_hkcVRM"

# 1> Choose LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-pro",
#     temperature=0.3
# )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
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
You are a strict product recommendation engine.

You must follow these rules:
1. Recommend ONLY products that appear in the provided product list.
2. If a budget is mentioned, recommend ONLY products within that budget.
3. Do NOT invent products.
4. Do NOT change prices.
5. If no product matches, clearly say "No matching products found in dataset."

User requirement:
{query}

Available products:
{context}

Now give the final recommendations in bullet points with price and reason.
"""


# 10> Generate Response

response = llm.invoke(prompt)
print(response.content)
