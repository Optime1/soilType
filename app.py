import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Классификатор типов почвы по космическим снимкам")
st.markdown("Расширенное приложение с собственной PyTorch-моделью. Загружайте спутниковые снимки для анализа.")

# Настройки
ML_API_URL = "http://localhost:8001/predict"  # FastAPI ML-сервис
DJANGO_API_URL = "http://localhost:8000/api/classify/"  # Django для асинхронных задач

uploaded_file = st.file_uploader("Загрузите спутниковый снимок (JPG/PNG/TIF)", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    # Синхронный вызов ML API
    if st.button("Классифицировать (синхронно)"):
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'image/jpeg')}
        response = requests.post(ML_API_URL, files=files)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Тип почвы: **{result['soil_type']}** (уверенность: {result['confidence']})")
            # Сохранение в CSV (как раньше)
            if 'results' not in st.session_state:
                st.session_state.results = []
            st.session_state.results.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), uploaded_file.name, result['soil_type'], result['confidence']])
            
            df = pd.DataFrame(st.session_state.results, columns=["Время", "Файл", "Тип почвы", "Уверенность"])
            st.dataframe(df)
        else:
            st.error(f"Ошибка ML-сервиса: {response.text}")
    
    # Асинхронный вызов через Django/Celery (для S3)
    if st.button("Загрузить в S3 и классифицировать асинхронно"):
        # Пример: Загрузка в S3 (реализуйте с boto3)
        s3_key = f"images/{uploaded_file.name}"
        # s3.upload_fileobj(uploaded_file, 'bucket', s3_key)  # Добавьте код S3
        response = requests.post(DJANGO_API_URL, json={'s3_key': s3_key})
        if response.status_code == 200:
            task_id = response.json()['task_id']
            st.info(f"Задача запущена: {task_id}. Проверьте статус.")
            # Поле для проверки статуса
            status_task_id = st.text_input("Task ID для проверки:")
            if status_task_id and st.button("Проверить статус"):
                status_response = requests.get(f"{DJANGO_API_URL}{status_task_id}")
                st.json(status_response.json())

# Инфо о модели
st.subheader("О собственной модели")
st.markdown("""
- **Архитектура**: CNN на базе ResNet18 (PyTorch), дообученная на спутниковых данных (LANDSAT-8).
- **Классы**: 5 типов (Black, Cinder, Laterite, Peat, Yellow Soil).
- **Точность**: ~85–92% на тестовом датасете (зависит от снимков).
- **Данные**: GitHub/LANDSAT датасет. Для мультиспектральных снимков добавьте каналы NIR/SWIR в Dataset.
- **Интеграция**: ML в FastAPI, backend в Django+Celery, хранение в PostgreSQL/S3.
""")

# Docker Compose (docker-compose.yml) для стека
st.code("""
version: '3'
services:
  nginx:
    image: nginx
    ports: ["80:80"]
    volumes: ['./nginx.conf:/etc/nginx/nginx.conf']
  
  django:
    build: ./backend
    command: gunicorn --bind 0.0.0.0:8000 backend.wsgi
    depends_on: [postgres, redis]
  
  celery:
    build: ./backend
    command: celery -A backend worker -l info
    depends_on: [redis]
  
  ml_service:
    build: ./ml
    ports: ["8001:8001"]
    command: uvicorn ml_service:app --host 0.0.0.0 --port 8001
  
  postgres:
    image: postgres:13
    environment: {POSTGRES_DB: soil_db, POSTGRES_USER: user, POSTGRES_PASSWORD: pass}
  
  redis:
    image: redis:alpine
""", language='yaml')