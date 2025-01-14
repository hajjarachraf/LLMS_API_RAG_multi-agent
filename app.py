# app.py
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Import the RAG system from previous code
# Assuming it's in rag_system.py
from rag_system import RAGSystem, RAGConfig

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize RAG system
config = RAGConfig()
rag_system = RAGSystem(config)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    answer, sources = rag_system.get_response(question)
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True)