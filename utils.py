import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text.strip()

def generate_embeddings(text, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    return sentences, embeddings

def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
