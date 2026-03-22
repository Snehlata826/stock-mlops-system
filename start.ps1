cd D:\stock-mlops-system
docker compose up backend mlflow -d
Start-Sleep -Seconds 8
ngrok http --domain=unpathetically-saturable-leandra.ngrok-free.dev 8000
