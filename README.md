# AI-Powered Resume Screener

This project is an AI-based resume screener built using NLP and machine learning techniques. It helps screen resumes based on their similarity to a job description, and it extracts key information such as the candidate's name and phone number using Named Entity Recognition (NER).

## Features

- **Text Cleaning**: Preprocessing of resumes by removing stop words, lemmatization, and stemming.
- **Similarity Measurement**: Uses TF-IDF and Word2Vec to measure the similarity between the resume and the job description.
- **Entity Extraction**: Extracts candidate name and phone number from the resumes.
- **Ranking**: Ranks the resumes based on their similarity to the job description.

## Setup

To run this project, you will need to install the necessary dependencies. You can do so by running:

```bash
pip install -r requirements.txt

