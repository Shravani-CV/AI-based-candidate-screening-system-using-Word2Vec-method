import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Mock interview data
data = {
    'candidate': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'response': [
        "I have experience in deep learning and natural language processing. I have worked on several projects involving neural networks.",
        "My expertise lies in machine learning algorithms and I have implemented various supervised and unsupervised techniques.",
        "I am familiar with computer vision and have developed image processing applications using TensorFlow.",
        "I have a strong background in AI research and have published papers on reinforcement learning.",
        "I have worked on big data technologies and have experience with Spark and Hadoop for data processing."
    ]
}

# Create DataFrame
df_candidates = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Apply preprocessing
df_candidates['processed_response'] = df_candidates['response'].apply(preprocess_text)

# Train Word2Vec model
model = Word2Vec(sentences=df_candidates['processed_response'], vector_size=100, window=5, min_count=1, sg=1)

# Vectorization function
def vectorize_response(response):
    vectors = [model.wv[word] for word in response if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Vectorize responses
df_candidates['response_vector'] = df_candidates['processed_response'].apply(vectorize_response)

# Define ideal response
ideal_response = "I have experience in deep learning and have worked on various neural network projects."
ideal_vector = vectorize_response(preprocess_text(ideal_response))

# Calculate similarity
df_candidates['similarity'] = df_candidates['response_vector'].apply(lambda x: cosine_similarity([x], [ideal_vector])[0][0])

# Rank candidates
df_candidates_sorted = df_candidates.sort_values(by='similarity', ascending=False)

# Display ranked candidates
print("Ranked Candidates based on similarity to ideal response:")
print(df_candidates_sorted[['candidate', 'similarity']])
