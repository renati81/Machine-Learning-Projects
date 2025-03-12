import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load IMDb Dataset
@st.cache_data
def load_data():
    file_path = r"C:\Users\HP\OneDrive\Desktop\UTA Docs\Spring 2025\CSE 6363 - Machine Learning\Assignments\Sematic_Search\IMDB Dataset for ML.csv"
    df = pd.read_csv(file_path)
    df['text'] = df['Title'] + ' ' + df['Overview'] + ' ' + df['Genre']
    df['text'] = df['text'].fillna('')
    return df

df = load_data()

# Load pre-trained sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Generate embeddings
@st.cache_resource
def create_faiss_index():
    embeddings = np.array(df['text'].apply(lambda x: model.encode(x)).tolist(), dtype='float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
    index.add(embeddings)
    return index, embeddings

index, embeddings = create_faiss_index()

# Function to perform a semantic search with cosine similarity
def search_movies(query, top_k=5):
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)  # Normalize query embedding
    _, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]]
    return results[['Title', 'Overview', 'Genre']]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie description, and we'll find similar movies for you!")

# Input field for user query
user_query = st.text_input("Describe the type of movie you'd like to watch:")

# Display results when user enters a query
if user_query:
    recommended_movies = search_movies(user_query, top_k=5)
    st.subheader("🔍 Recommended Movies:")
    
    for idx, row in recommended_movies.iterrows():
        st.markdown(f"**🎥 {row['Title']}**")
        st.write(f"📖 *{row['Overview']}*")
        st.write(f"🎭 Genre: {row['Genre']}")
        st.markdown("---")
