import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import nltk

# --- Tell NLTK where to find the data ---
nltk.data.path.append(os.path.expanduser('~/nltk_data'))

# --- Initial Setup ---
load_dotenv()
CORPUS_DIR = "./corpus/"
DB_DIR = "./musk_db/"

class MuskTwinV2:
    """A sophisticated AI Digital Twin of Elon Musk with an advanced RAG pipeline."""
    
    def __init__(self):
        """Initializes the RAG pipeline, building the vector store if it doesn't exist."""
        print("🧠 Initializing AI Core...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # For TTS
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if not os.path.exists(DB_DIR):
            print("   No existing vector store found. Building a new one from the corpus...")
            self._create_and_persist_db()
        
        print("   Loading vector store...")
        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        print("   Vector store loaded successfully.")
        
        self._setup_rag_chain()
        print("✅ AI Core initialized successfully.")

    def _create_and_persist_db(self):
        """Loads data from the corpus, splits it, and creates the Chroma vector store."""
        all_documents = []
        file_paths = [os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')]
        
        if not file_paths:
            raise ValueError("Corpus directory is empty. Run data_collector_v2.py first.")

        print(f"   Found {len(file_paths)} document(s) in corpus.")
        for file_path in file_paths:
            print(f"      Loading: {os.path.basename(file_path)}")
            loader = TextLoader(file_path, encoding='utf-8')
            all_documents.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)
        
        print(f"   Splitting documents into {len(splits)} chunks.")
        print("   Creating and persisting vector store... This may take a few moments.")
        
        Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=DB_DIR
        )
        print("   ✅ Vector store created and persisted.")

    def _setup_rag_chain(self):
        """Defines and sets up the LangChain RAG pipeline."""
  
  
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        template = """
        You are an AI emulating Elon Musk. Answer the user's question based on the provided context.
        Your persona should be direct, concise, and focused on engineering and first principles.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.3)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        self.retriever = retriever
        
    def generate_audio(self, text: str) -> bytes:
        """Generates audio from text using OpenAI's TTS API."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=text
            )
            return response.content
        except Exception as e:
            print(f"🔴 Error generating audio: {e}")
            return None

    def ask(self, question: str, chat_history: list = None) -> dict:
        """Asks a question to the digital twin and generates audio for the answer."""
        if not question:
            return {"answer": "Please ask a question.", "sources": [], "context_chunks": [], "audio": None}

        retrieved_docs = self.retriever.get_relevant_documents(question)
        sources = list(set([os.path.basename(doc.metadata.get("source", "")) for doc in retrieved_docs]))
        context_chunks = [doc.page_content for doc in retrieved_docs]
        
        answer = self.chain.invoke(question)
        
        audio_content = self.generate_audio(answer)
        
        return {"answer": answer, "sources": sources, "context_chunks": context_chunks, "audio": audio_content}

if __name__ == "__main__":
    print("--- Running AI Core Setup ---")
    twin = MuskTwinV2()
    print("\n--- Testing audio generation ---")
    test_audio = twin.generate_audio("Testing text to speech.")
    if test_audio:
        with open("test_audio.mp3", "wb") as f:
            f.write(test_audio)
        print("✅ Test audio saved to test_audio.mp3")
    else:
        print("🔴 Test audio generation failed.")
    
    print("\n--- AI Core Setup Complete ---")
