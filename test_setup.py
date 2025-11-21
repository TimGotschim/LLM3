# test_setup.py
import os
import requests

# Test 1: Check if PDF folder exists
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "rexx_pdfs")
if os.path.exists(pdf_folder):
    pdf_count = len([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')])
    print(f"✅ Found {pdf_count} PDFs in {pdf_folder}")
else:
    print(f"❌ PDF folder not found at {pdf_folder}")

# Test 2: Check if packages are installed
try:
    import chromadb
    print("✅ ChromaDB installed")
except ImportError:
    print("❌ ChromaDB not installed")

try:
    import sentence_transformers
    print("✅ Sentence Transformers installed")
except ImportError:
    print("❌ Sentence Transformers not installed")

# Test 3: Check if Ollama is running
try:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"✅ Ollama is running with {len(models)} models")
        for model in models:
            print(f"   - {model['name']}")
    else:
        print("❌ Ollama is not responding correctly")
except:
    print("❌ Cannot connect to Ollama - make sure it's running")