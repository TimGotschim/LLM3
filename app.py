"""
Web UI for Rexx RAG System
Run with: python app.py
Then open http://localhost:5000 in your browser
"""

import os
from flask import Flask, render_template, request, jsonify
from rexx_rag_system import RexxRAGSystem
import time

app = Flask(__name__)

# Initialize RAG system
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "rexx_pdfs")

print("Initializing RAG system...")
rag_system = RexxRAGSystem(pdf_folder)
rag_system.process_pdfs()
print("RAG system ready!")


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """Handle question queries."""
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        start_time = time.time()
        response = rag_system.query(question, n_results=5)
        response_time = time.time() - start_time

        # Format sources for display (use retrieved_chunks which has full info)
        sources = []
        for chunk in response.get('retrieved_chunks', []):
            text = chunk.get('text', '')
            sources.append({
                'text': text[:500] + '...' if len(text) > 500 else text,
                'source': chunk.get('source', 'Unknown'),
                'distance': round(chunk.get('distance', 0), 4)
            })

        return jsonify({
            'answer': response.get('answer', 'No answer generated'),
            'sources': sources,
            'response_time': round(response_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate an answer against expected answer."""
    data = request.json
    answer = data.get('answer', '')
    expected = data.get('expected', '')

    if not answer or not expected:
        return jsonify({'error': 'Both answer and expected answer required'}), 400

    # Simple evaluation metrics
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # TF-IDF Cosine Similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([answer.lower(), expected.lower()])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Keyword overlap
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())
        overlap = len(answer_words & expected_words) / max(len(expected_words), 1)

        # Contains check
        contains_expected = expected.lower() in answer.lower()

        return jsonify({
            'similarity_score': round(similarity * 100, 1),
            'keyword_overlap': round(overlap * 100, 1),
            'contains_expected': contains_expected,
            'overall_score': round((similarity * 0.6 + overlap * 0.4) * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get system statistics."""
    try:
        stats = rag_system.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
