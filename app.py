from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import glob

app = Flask(__name__)

# Variáveis globais para os dados
df_merged = None
scaler = None
model = None

def load_training_data_antigo(train_dir):
    train_df = pd.DataFrame()
    for filename in os.listdir(train_dir):
        if filename.startswith('treino_parte') and filename.endswith('.csv'):
            file_path = os.path.join(train_dir, filename)
            df = pd.read_csv(file_path)
            train_df = pd.concat([train_df, df], ignore_index=True)
    return train_df

def load_items_data_antigo(items_dir):
    items_df = pd.DataFrame()
    for filename in os.listdir(items_dir):
        if filename.startswith('itens-parte') and filename.endswith('.csv'):
            file_path = os.path.join(items_dir, filename)
            df = pd.read_csv(file_path)
            items_df = pd.concat([items_df, df], ignore_index=True)
    return items_df

def carregar_dados_treino(diretorio_treino):
    arquivos_treino = glob.glob(os.path.join(diretorio_treino, '*.csv'))
    treino_df = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos_treino), ignore_index=True)
    return treino_df

def carregar_dados_itens(diretorio_itens):
    arquivos_itens = glob.glob(os.path.join(diretorio_itens, '*.csv'))
    itens_df = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos_itens), ignore_index=True)
    return itens_df

def initialize_data():
    """Carrega e processa os dados de treino e itens."""
    global df_merged
    train_dir = 'arquivos_divididos_treino/'
    items_dir = 'arquivos_divididos_itens/'

    items_df = carregar_dados_itens(items_dir)
    train_df = carregar_dados_treino(train_dir)
    train_df = train_df[train_df['userType'] == 'Logged']

    # Processar colunas
    columns_to_explode = ['history', "scrollPercentageHistory", "timeOnPageHistory", 
                          'timestampHistory', 'pageVisitsCountHistory', "numberOfClicksHistory", 'timestampHistory_new']
    for col in columns_to_explode:
        train_df[col] = train_df[col].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    df_exploded = train_df.explode(columns_to_explode, ignore_index=True)
    df_merged = pd.merge(df_exploded, items_df, left_on="history", right_on="page", how="left").drop(columns="history")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    userId = request.form['userId']
    limit = int(request.form.get('limit', 10))  # Limite de previsões
    return make_prediction(userId, limit)

@app.route('/random_predict', methods=['POST'])
def random_predict():
    if df_merged is not None:
        random_userId = random.choice(df_merged['userId'].unique())
        limit = int(request.form.get('limit', 10))  # Limite de previsões
        return make_prediction(random_userId, limit)
    return render_template('index.html', error="Erro ao gerar UserId aleatório.")

def make_prediction(userId, limit):
    user_data = df_merged[df_merged['userId'] == userId]

    if user_data.empty:
        return render_template('index.html', userId=userId, futuros_acessos=["Sem dados suficientes"])

    user_data = user_data.sort_values(by='timestampHistory', ascending=True)
    user_data['next_page'] = user_data['page'].shift(-1)  # Próxima página como alvo
    user_data = user_data.dropna(subset=['next_page'])

    pages = user_data['page'].unique()
    for page in pages:
        user_data[f'visited_{page}'] = user_data['page'].apply(lambda x: 1 if x == page else 0)

    X = user_data[['historySize', 'timeOnPageHistory', 'numberOfClicksHistory', 
                   'scrollPercentageHistory'] + [f'visited_{page}' for page in pages]]
    y = user_data['next_page']

    if X.empty:
        return render_template('index.html', userId=userId, futuros_acessos=["Sem dados suficientes"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)
    return render_template('index.html', userId=userId, futuros_acessos=list(predictions[:limit]))

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
