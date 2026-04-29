FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY app_hf.py app.py
COPY dataset.csv .
COPY train_model.py .
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]