from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Conectar con MongoDB
client = MongoClient("mongodb+srv://admin:admin@cluster0.57bk3.mongodb.net/")
db = client['RecomendationSystem_db']
movies_collection = db['movies']
movie_details_collection = db['movie_details']

# Inicializar la app de FastAPI
app = FastAPI()

# Modelo de datos de entrada
class MovieRequest(BaseModel):
    title: str
    num_recommendations: int = 5

# Ruta de prueba
@app.get("/")
def read_root():
    return {"message": "API de Recomendación de Películas Activa"}

# Ruta para obtener recomendaciones
@app.post("/recommend")
def recommend_movies(request: MovieRequest):
    # Cargar datos desde MongoDB
    movies_data = list(movies_collection.find())
    df = pd.DataFrame(movies_data)

    # Vectorización y cálculo de similitudes
    if "overview" not in df.columns:
        raise HTTPException(status_code=500, detail="Column 'overview' not found in data.")
    df["overview"] = df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Buscar la película y recomendaciones
    try:
        idx = df[df["title"] == request.title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:request.num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = df.iloc[movie_indices]["title"].tolist()
        return {"recommendations": recommended_movies}
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")

