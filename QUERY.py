import os
from rexx_rag_system import RexxRAGSystem

# Initialize the system - use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "rexx_pdfs")
rag_system = RexxRAGSystem(pdf_folder)

# Process PDFs (only needed first time)
rag_system.process_pdfs()

# Ask your question
response = rag_system.query("Can two users have the same User Name in Rexx?")
print(response["answer"])