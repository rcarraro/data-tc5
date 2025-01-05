# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie o arquivo requirements.txt para o container
COPY requirements.txt .

# Instale os pacotes listados no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código para o container
COPY . .

# Exponha a porta que o Flask utilizará
EXPOSE 5000

# Comando para iniciar o aplicativo Flask
CMD ["python", "app.py"]
