#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import requests
import os

# ------------------------------------------------------------
# 1. Configuración: URLs de los artefactos en GitHub
# ------------------------------------------------------------
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/CrisSCon/MIAD_ML_NLP_2026_Disponibilizacion1/main"

MODEL_URL = f"{GITHUB_RAW_BASE}/model.pkl"
FEATURES_URL = f"{GITHUB_RAW_BASE}/features.pkl"
ENCODINGS_URL = f"{GITHUB_RAW_BASE}/target_encodings.pkl"

# ------------------------------------------------------------
# 2. Función para descargar archivos si no existen localmente
# ------------------------------------------------------------
def download_file(url, local_path):
    """Descarga un archivo desde una URL si no existe localmente."""
    if not os.path.exists(local_path):
        print(f"⬇️  Descargando {local_path} desde GitHub...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"✅ {local_path} descargado correctamente.")
    else:
        print(f"📁 Usando archivo local: {local_path}")

# ------------------------------------------------------------
# 3. Descargar y cargar los artefactos
# ------------------------------------------------------------
download_file(MODEL_URL, "model.pkl")
download_file(FEATURES_URL, "features.pkl")
download_file(ENCODINGS_URL, "target_encodings.pkl")

model = joblib.load("model.pkl")
FEATURE_NAMES = joblib.load("features.pkl")
target_encodings = joblib.load("target_encodings.pkl")
global_mean = target_encodings["global_mean"]

print("🚀 Modelo y artefactos cargados. API lista para recibir peticiones.\n")

# ------------------------------------------------------------
# 4. Funciones de feature engineering (idénticas al entrenamiento)
# ------------------------------------------------------------
def create_raw_features(row):
    """Crea las variables derivadas: log, interacciones, etc."""
    d = row.copy()
    d['durlog'] = np.log1p(d['duration_ms'])
    d['eneloud'] = d['energy'] * d['loudness']
    d['danval'] = d['danceability'] * d['valence']
    d['isinstr'] = int(d['instrumentalness'] > 0.45)
    d['expl'] = int(d['explicit'])
    return d

def add_target_encodings(row):
    """Agrega las columnas de target encoding con suavizado global."""
    artist = row.get('artists', '')
    genre = row.get('track_genre', '')
    album = row.get('album_name', '')
    row['artistste'] = target_encodings['artists'].get(artist, global_mean)
    row['track_genrete'] = target_encodings['track_genre'].get(genre, global_mean)
    row['album_namete'] = target_encodings['album_name'].get(album, global_mean)
    return row

# ------------------------------------------------------------
# 5. Configuración de Flask‑RESTX
# ------------------------------------------------------------
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Spotify Popularity API',
    description='Predicción de popularidad de canciones (Stacking XGB + LGBM)'
)

ns = api.namespace('predict', description='Predicción de popularidad')

# Parser (usando ns.parser() para que Swagger lo documente)
parser = ns.parser()
numeric_params = [
    'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]
for param in numeric_params:
    parser.add_argument(param, type=float, required=True, location='args')

parser.add_argument('artists', type=str, required=True, location='args')
parser.add_argument('track_genre', type=str, required=True, location='args')
parser.add_argument('album_name', type=str, required=True, location='args')

# Modelo de respuesta para Swagger
resource_fields = api.model('Resource', {
    'popularity': fields.Float,
})

# ------------------------------------------------------------
# 6. Endpoint
# ------------------------------------------------------------
@ns.route('/')
class PredictPopularity(Resource):

    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        """Predicción vía GET con parámetros en la URL"""
        args = parser.parse_args()
        return self._predict(args)

    def post(self):
        """Predicción vía POST con JSON en el cuerpo"""
        data = request.get_json()
        return self._predict(data)

    def _predict(self, args):
        # Convertir argumentos a diccionario
        row = {k: v for k, v in args.items()}
        # Aplicar feature engineering
        row = create_raw_features(row)
        row = add_target_encodings(row)
        # Construir array en el orden exacto que espera el modelo
        X = np.array([row[name] for name in FEATURE_NAMES]).reshape(1, -1)
        # Predecir y limitar al rango [0, 100]
        pred = model.predict(X)[0]
        pred = float(np.clip(pred, 0, 100))
        return {"popularity": pred}, 200

# ------------------------------------------------------------
# 7. Lanzar la aplicación
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:




