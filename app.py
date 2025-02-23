from flask import Flask, render_template, request
import os
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import glob

app = Flask(__name__)

df_merged = None
scaler = StandardScaler()

def carregar_dados(diretorio):
    arquivos = glob.glob(os.path.join(diretorio, '*.csv'))
    return pd.concat((pd.read_csv(arquivo) for arquivo in arquivos), ignore_index=True)

def initialize_data():
    """Carrega e processa os dados de treino e itens."""
    global df_merged
    train_dir = 'arquivos_divididos_treino/'
    items_dir = 'arquivos_divididos_itens/'

    items_df = carregar_dados(items_dir)
    train_df = carregar_dados(train_dir)
    train_df = train_df[train_df['userType'] == 'Logged']

    columns_to_explode = ['history', 'scrollPercentageHistory', 'timeOnPageHistory', 
                          'timestampHistory', 'pageVisitsCountHistory', 'numberOfClicksHistory']
    for col in columns_to_explode:
        train_df[col] = train_df[col].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    df_exploded = train_df.explode(columns_to_explode, ignore_index=True)
    df_merged = pd.merge(df_exploded, items_df, left_on='history', right_on='page', how='left').drop(columns='history')

@app.route('/')
def index():
    return render_template('index.html')

def make_prediction(userId, limit, modelo):
    user_data = df_merged[df_merged['userId'] == userId]
    if user_data.empty:
        return render_template('index.html', userId=userId, futuros_acessos=[])

    resultados = []

    if modelo in ['random_forest', 'both']:
        user_data = user_data.sort_values(by='timestampHistory', ascending=True)
        user_data['next_page'] = user_data['page'].shift(-1)
        user_data = user_data.dropna(subset=['next_page'])

        X = user_data[['timeOnPageHistory', 'numberOfClicksHistory', 'scrollPercentageHistory']]
        y = user_data['next_page']
        
    if modelo in ['random_forest', 'both']:
        user_data = user_data.sort_values(by='timestampHistory', ascending=True)
        user_data['next_page'] = user_data['page'].shift(-1)
        user_data = user_data.dropna(subset=['next_page'])

        X = user_data[['timeOnPageHistory', 'numberOfClicksHistory', 'scrollPercentageHistory']]
        y = user_data['next_page']
        
        if not X.empty:
            X_scaled = scaler.fit_transform(X)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            predictions = model.predict(X_scaled)
            resultados.append({"modelo": "Random Forest", "acessos": list(predictions[:limit])})

    if modelo in ['kmeans', 'both']:
        X = user_data[['timeOnPageHistory', 'numberOfClicksHistory', 'scrollPercentageHistory']]
        if not X.empty:
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=min(len(X), 5), random_state=42, n_init=10)
            user_data['cluster'] = kmeans.fit_predict(X_scaled)
            df_merged.loc[user_data.index, 'cluster'] = user_data['cluster']
            cluster = user_data.iloc[-1]['cluster']
            common_pages = df_merged[df_merged['cluster'] == cluster]['page'].value_counts().index[:limit]
            resultados.append({"modelo": "K-Means", "acessos": list(common_pages)})
    
    return render_template('index.html', userId=userId, futuros_acessos=resultados)

@app.route('/predict', methods=['POST'])
def predict():
    userId = request.form['userId']
    if userId == 'random':
        userId = random.choice(df_merged['userId'].unique())
    limit = int(request.form.get('limit', 10))
    return make_prediction(userId, limit, request.form.get('modelo'))

@app.route('/predict_all', methods=['POST'])
def predict_all():
    userId = request.form['userId']
    if userId == 'random':
        userId = random.choice(df_merged['userId'].unique())
    limit = int(request.form.get('limit', 10))
    return make_prediction(userId, limit,request.form.get('modelo'))

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, host='0.0.0.0', port=5000)