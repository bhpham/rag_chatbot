# rag_chatbot
RAG Customer Chatbot

Installation Guide:
* Open terminal and run the following:
  1. uv venv .venv --python 3.11 && source .venv/bin/activate
  2. uv pip install -r requirements.txt
  3. python3 -m ipykernel install --user --name=rag_chatbot --display-name "rag_chatbot"
  4. ollama pull gemma3:1b
  5. streamlit run app.py 
