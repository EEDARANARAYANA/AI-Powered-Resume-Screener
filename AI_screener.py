import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import docx2txt
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

#Dowload required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

#Load spacy English model for lemmatization
nlp=spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Extract text from file
def extract_text_from_file(file):
    if file.name.endswith('.pdf'):
        return extract_text(file)
    elif file.name.endswith('.docx'):
        return docx2txt.process(file)
    else:
        return ""

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    stemmed = [stemmer.stem(word) for word in lemmatized]
    return " ".join(stemmed)

# Extract name and phone number
def extract_name_and_phone(text):
    doc = nlp(text)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    phone = re.search(r'(\+?\d{1,3}[\s-]?)?(\(?\d{3,4}\)?[\s-]?)?\d{3}[\s-]?\d{4}', text)
    phone_number = phone.group() if phone else None
    return name, phone_number

# TF-IDF Similarity
def compute_tfidf_similarity(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    documents = [job_description] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    return similarity_scores.flatten()

# Word2Vec Similarity
def compute_word2vec_similarity(resume_texts, job_description):
    all_texts = [text.split() for text in resume_texts] + [job_description.split()]
    model = Word2Vec(sentences=all_texts, vector_size=100, window=5, min_count=1, workers=4)

    def avg_vector(text):
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

    job_vec = avg_vector(job_description)
    resume_vectors = [avg_vector(text) for text in resume_texts]
    similarity_scores = [cosine_similarity([job_vec], [vec])[0][0] for vec in resume_vectors]
    return similarity_scores

# Streamlit app
st.title("AI-Based Resume Screener")
st.write("Upload resumes")

# Job Description
job_desc = st.text_area("Paste the Job Description:", height=500)

# Resumes Upload
uploaded_files = st.file_uploader("Upload Resume PDFs", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("Screen Resumes"):
    if job_desc and uploaded_files:
        raw_texts = [extract_text_from_file(file) for file in uploaded_files]
        resume_texts = [clean_text(text) for text in raw_texts]
        job_description = clean_text(job_desc)

        # Compute similarity scores
        tfidf_scores = compute_tfidf_similarity(resume_texts, job_description)
        word2vec_scores = compute_word2vec_similarity(resume_texts, job_description)

        # NER extraction
        extracted_info = [extract_name_and_phone(text) for text in raw_texts]
        names = [info[0] for info in extracted_info]
        phone = [info[1] for info in extracted_info]

        # Combine scores with filenames
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Candidate Name": names,
            "Phone Number": phone,
            "TF-IDF Similarity": tfidf_scores,
            "Word2Vec Similarity": word2vec_scores,
            "Final Score": (tfidf_scores + word2vec_scores) / 2
        })

        results = results.sort_values(by="Final Score", ascending=False)

        st.subheader("Resume Ranking")
        st.dataframe(results)

    else:
        st.warning("⚠️ Please upload resumes and enter a job description.")
