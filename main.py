from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings





import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the variable
password = os.getenv("GOOGLE_API_KEY")


pdf_folder = r"C:\Users\DELL\Documents\Research Papers"

loader = PyPDFLoader(pdf_folder)

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

splits = text_splitter.split_documents(document)

print(f"documents split into {len(splits)}")


embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
vector_store = Chroma.from_documents(splits, embedding)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=password)

system_prompt = ("You are an expert Q&A assistant. Answer the user's question "
    "based *only* on the provided context. If the context does not "
    "contain the answer, clearly state that you cannot find the answer "
    "in the provided documents."
    "\n\n"
    "{context}")


prompt = ChatPromptTemplate(
    [("system", system_prompt),
     ("human", "{question}")]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever|format_docs,
        "question": RunnablePassthrough()
    }
    |prompt
    |llm
    |StrOutputParser()
)
question = "what is a stateful tool"
# Instead of response = rag_chain.invoke(question)
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)