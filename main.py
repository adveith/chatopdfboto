import os
import pickle
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

# Load the embeddings
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Find the closest sentence
def find_closest_sentence(question, embeddings, sentences, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = []
    for sentence, embedding in zip(sentences, embeddings):
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_embedding, sentence_embedding)[0][0].item()
        scores.append((sentence, score))
    closest_sentence = max(scores, key=lambda x: x[1])
    return closest_sentence

# Answer a question
def answer_question(question, model, embeddings, sentences):
    closest_sentence = find_closest_sentence(question, embeddings, sentences, model)
    return closest_sentence[0]

# Main execution
if __name__ == "__main__":
    embeddings_path = 'embeddings/document_embeddings.pkl'
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Check if embeddings file exists and load it
    if os.path.exists(embeddings_path):
        sentences, embeddings = load_embeddings(embeddings_path)
    else:
        # Generate embeddings for the first time
        with open('sample.pdf', 'rb') as f:
            text = extract_text_from_pdf(f)
        sentences = text.split('\n')
        embeddings = [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]
        with open(embeddings_path, 'wb') as f:
            pickle.dump((sentences, embeddings), f)

    # Interaction loop
    while True:
        question = input("Ask a question (or enter 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = answer_question(question, model, embeddings, sentences)
        print(f"Answer: {answer}")
