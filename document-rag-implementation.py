# --------------------------------------------------
# 100 % LangChain 1.0 compliant RAG chat-bot
# --------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama  # 1.0 partner package
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain  # FIXED IMPORT
from langchain.chains.combine_documents import create_stuff_documents_chain  # FIXED IMPORT
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model  # 1.0 helper (optional but future-proof)
import os


class DocumentQAChatbot:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.qa_chain = None

    # ------------------------------------------------------------------
    # 1.  Load PDF
    # ------------------------------------------------------------------
    def load_document(self):
        print(f"📄 Loading document: {self.pdf_path}")
        documents = PyPDFLoader(self.pdf_path).load()
        print(f"✅ Loaded {len(documents)} pages")
        return documents

    # ------------------------------------------------------------------
    # 2.  Chunk
    # ------------------------------------------------------------------
    def split_documents(self, documents):
        print("✂️  Splitting document into chunks…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # 3.  Embed → FAISS
    # ------------------------------------------------------------------
    def create_vectorstore(self, chunks):
        print("🔮 Creating embeddings and vector store…")
        embeddings = OllamaEmbeddings(
            model="llama3.1:8b", base_url="http://localhost:11434"
        )
        try:
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            print("✅ Vector store ready")
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            print("   → Is Ollama running and is llama3.1:8b pulled?")
            exit(1)

    # ------------------------------------------------------------------
    # 4.  Build 1.0 LCEL chain
    # ------------------------------------------------------------------
    def setup_qa_chain(self):
        print("🔗 Building 1.0 LCEL retrieval chain…")

        # --- LLM (future-proof helper) ---
        llm = init_chat_model(
            "llama3.1:8b",
            model_provider="ollama",
            temperature=0.3,
            base_url="http://localhost:11434",
        )

        # --- Prompt ---
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise (≤3 sentences).\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )

        # --- Chains ---
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        print("✅ QA chain ready")

    # ------------------------------------------------------------------
    # 5.  One-shot initialisation
    # ------------------------------------------------------------------
    def initialize(self):
        print("\n" + "=" * 60)
        print("🤖 Initialising Document Q&A Chat-bot (LangChain 1.0)")
        print("=" * 60 + "\n")
        docs = self.load_document()
        chunks = self.split_documents(docs)
        self.create_vectorstore(chunks)
        self.setup_qa_chain()
        print("\n" + "=" * 60)
        print("✅ Chat-bot ready – ask your questions!")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 6.  Single question
    # ------------------------------------------------------------------
    def ask(self, question: str):
        if not self.qa_chain:
            raise RuntimeError("Chat-bot not initialised. Run .initialize() first.")

        print(f"\n🤔 Question: {question}\n💭 Thinking…\n")
        response = self.qa_chain.invoke({"input": question})
        answer = response.get("answer", "No answer found.")
        sources = response.get("context", [])

        print("💡 Answer:", answer)
        if sources:
            pages = {doc.metadata.get("page", "unknown") for doc in sources}
            print(f"📚 Sources: {len(sources)} chunks  |  Pages: {sorted(pages)}")
        return answer

    # ------------------------------------------------------------------
    # 7.  Interactive loop
    # ------------------------------------------------------------------
    def chat(self):
        print("\n💬 Chat mode  –  type 'exit'/'q' to quit")
        print("-" * 60)
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            try:
                self.ask(question)
            except Exception as e:
                print(f"❌ Error: {e}")


# ----------------------------------------------------------------------
# Run only when executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    PDF_PATH = "your_document.pdf"  # <— change to your file
    if not os.path.isfile(PDF_PATH):
        print(f"❌ File not found: {PDF_PATH}")
        exit(1)

    bot = DocumentQAChatbot(PDF_PATH)
    bot.initialize()
    bot.chat()