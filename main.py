import os
import socket
import sys
from dotenv import load_dotenv
import httpx
from google import genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
API_HOST = "generativelanguage.googleapis.com"


def require_api_key() -> str:
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY in your .env file. GOOGLE_API_KEY is also accepted for backward compatibility."
        )
    return api_key


def check_google_dns(host: str = API_HOST) -> None:
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"DNS lookup failed for {host}. Your machine cannot currently resolve Google's Gemini API host. "
            "Check your internet connection, DNS settings, VPN/proxy, or firewall and try again."
        ) from exc


def run_connectivity_preflight(host: str = API_HOST) -> None:
    print(f"Using Gemini API host: {host}")
    check_google_dns(host)
    print("Connectivity preflight passed: Gemini API hostname resolved successfully.")


def explain_connection_error(exc: Exception) -> str:
    message = str(exc)
    if "getaddrinfo failed" in message:
        return (
            f"Network error: your machine could not resolve {API_HOST}. "
            "This is usually a DNS, proxy, VPN, firewall, or internet connectivity issue."
        )
    return f"Network error while contacting Gemini API: {message}"

# Custom embedding wrapper using google-genai SDK directly
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text]
        )
        return result.embeddings[0].values

# 1. CONFIGURATION
pdf_folder_path = r"C:\Users\DELL\Documents\Research Papers"
persist_directory = "./chroma_db_research"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_vector_store(embedding: Embeddings) -> Chroma:
    if os.path.exists(persist_directory):
        print("Loading existing vector store from disk...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

    if not os.path.isdir(pdf_folder_path):
        raise RuntimeError(f"PDF folder not found: {pdf_folder_path}")

    print("No existing database found. Starting ingestion...")
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print("Ingestion complete. Database saved.")
    return vector_store


def main() -> None:
    checked_api_key = require_api_key()
    run_connectivity_preflight()

    embedding = GeminiEmbeddings(api_key=checked_api_key)
    vector_store = build_vector_store(embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=checked_api_key
    )

    system_prompt = (
        "You are an expert Q&A assistant. Answer the user's question "
        "based *only* on the provided context. If the context does not "
        "contain the answer, clearly state that you cannot find the answer "
        "in the provided documents.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "Capturing biological structure in a latent low-dimensional space"
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError as exc:
        print(explain_connection_error(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
