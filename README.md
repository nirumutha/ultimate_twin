🚀 AI Digital Twin: Elon Musk
This project is a sophisticated, multi-modal AI Digital Twin of Elon Musk, built using a Retrieval-Augmented Generation (RAG) pipeline. It ingests public interviews and writings to answer questions in his specific voice and style, providing both text and audio responses.

This application serves as a comprehensive portfolio piece demonstrating a full-cycle AI engineering project, from resilient data collection to a polished, interactive web application.

Live Demo URL: https://nirajmutha-musk-twin.streamlit.app/

✨ Key Features & Solutions
This project goes beyond a standard RAG implementation by incorporating several features designed to solve real-world engineering challenges.

1. Resilient & Scalable Data Collection
My initial data collection script was fragile; it would fail on long-running tasks or unavailable data. I re-engineered it to be resilient and scalable.

Progress Tracking: I integrated a SQLite database to track all processed URLs. This was crucial because if the script failed halfway through a long YouTube transcription, I didn't have to start over from scratch, saving significant time and API costs.

Automatic Audio Chunking: During testing, I hit a hard 25MB file size limit from the OpenAI Whisper API. To solve this, I engineered a pre-processing step using the pydub library to automatically chunk oversized audio files into API-compliant 10-minute segments before transcription.

Deployment Bug Fix: When deploying to Streamlit Cloud, the app crashed due to an outdated version of sqlite3 on the server. I solved this by adding pysqlite3-binary to the requirements and including a special override at the top of my app_v2.py file to force the app to use the newer version.

2. Multi-Modal Output
To create a more engaging user experience, the application generates both text and audio.

Text Generation: Uses GPT-4-turbo for high-quality, context-aware responses.

Audio Generation: Integrates OpenAI's Text-to-Speech (TTS) API to convert the text answer into spoken words, which can be played directly in the user interface.

3. AI Transparency & Explainability
To demystify the AI's process, I designed the UI for full transparency.

Source Citation: For every answer, the application clearly cites which source document(s) from the knowledge base were used as the primary context. This demonstrates a commitment to building "grounded" and verifiable AI systems.

🛠️ Tech Stack
AI Orchestration: LangChain

LLMs & APIs: OpenAI (GPT-4-turbo, Whisper, TTS, Embeddings)

Vector Database: ChromaDB

Web Framework: Streamlit

Data Collection: yt-dlp, trafilatura, pydub

Core Language: Python

⚙️ Setup and Installation
Clone the repository:

git clone https://github.com/nirumutha/ultimate_twin.git
cd ultimate_twin

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Set up your environment variables:

Create a .env file in the root directory.

Add your OpenAI API key: OPENAI_API_KEY="sk-..."

Run the application:

First, build the knowledge base: python data_collector_v2.py

 Then, build the vector database: python ai_core_v2.py

Finally, launch the app: streamlit run app_v2.py
