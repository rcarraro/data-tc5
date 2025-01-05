docker build -t flask-access-predictor .
docker run -d -p 5000:5000 --name flask-predict flask-access-predictor
