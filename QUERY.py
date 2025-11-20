from rexx_rag_system import RexxRAGSystem

# Initialize the system
pdf_folder = "/Users/timgotschim/Documents/LLM/Large-Language-Models/rexx_pdfs"
rag_system = RexxRAGSystem(pdf_folder)

# Process PDFs (only needed first time)
# rag_system.process_pdfs()

# Ask your question
response = rag_system.query("Can two users have the same User Name in Rexx?")
print(response["answer"])