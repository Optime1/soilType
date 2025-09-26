import streamlit as st
import pandas as pd
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import tempfile
import os
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='soil_classifier.log'
)
logger = logging.getLogger(__name__)

if 'results' not in st.session_state:
    st.session_state.results = []
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5

def save_results_to_csv(results, filename="soil_classification_results.csv"):
    """Сохраняет результаты классификации в CSV файл."""
    df = pd.DataFrame(results, columns=["Timestamp", "Image Name", "Soil Type", "Confidence"])
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")

def get_csv_download_link(df, filename="soil_classification_results.csv"):
    """Создаёт ссылку для скачивания CSV файла."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def plot_confidence_histogram(results):
    """Создаёт гистограмму уверенности классификации."""
    confidences = [float(r[3]) for r in results]
    if confidences:
        plt.figure(figsize=(8, 4))
        sns.histplot(confidences, bins=10, kde=True, color='blue')
        plt.title("Распределение уверенности модели")
        plt.xlabel("Уверенность (%)")
        plt.ylabel("Количество")
        st.pyplot(plt)

st.title("Классификатор типов почвы по фотографии")
st.markdown("""
    Это приложение позволяет загружать фотографии полей и определять тип почвы с помощью модели машинного обучения.
    Поддерживаются изображения в форматах JPG, JPEG, PNG. Результаты можно сохранить в CSV.
""")

with st.sidebar:
    st.header("Настройки модели")
    model_id = st.text_input(
        "ID модели Roboflow",
        value="soil-type-model/1",
        help="Введите ID модели из Roboflow Universe (например, 'soil-type-model/1')."
    )
    api_key = st.text_input(
        "Roboflow API Key",
        type="password",
        help="Введите ваш API-ключ Roboflow (получите на app.roboflow.com)."
    )
    confidence_threshold = st.slider(
        "Порог уверенности",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Минимальная уверенность для принятия результата."
    )
    st.session_state.confidence_threshold = confidence_threshold
    st.markdown("**Примечание**: Для точных результатов используйте чёткие фото почвы крупным планом.")

# Поле для загрузки изображений
st.subheader("Загрузка изображений")
uploaded_files = st.file_uploader(
    "Выберите одно или несколько фото почвы",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if api_key:
    try:
        CLIENT = InferenceHTTPClient(
            api_url="https://classify.roboflow.com",
            api_key=api_key
        )
        logger.info("Roboflow client initialized successfully.")
    except Exception as e:
        st.error(f"Ошибка инициализации клиента Roboflow: {str(e)}")
        logger.error(f"Failed to initialize Roboflow client: {str(e)}")
else:
    st.warning("Пожалуйста, введите API-ключ Roboflow в боковой панели.")
    logger.warning("No API key provided.")

if uploaded_files:
    st.subheader("Результаты классификации")
    results = st.session_state.results
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Загруженное фото: {uploaded_file.name}", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(tmp_file.name)
                image_path = tmp_file.name

            start_time = time.time()
            with st.spinner(f"Обработка {uploaded_file.name}..."):
                result = CLIENT.infer(image_path, model_id=model_id)
            
            os.remove(image_path)
            logger.info(f"Processed image: {uploaded_file.name}, time: {time.time() - start_time:.2f}s")

            if 'predictions' in result and result['predictions']:
                predictions = result['predictions']
                top_prediction = max(predictions, key=lambda x: x['confidence'])
                soil_type = top_prediction['class']
                confidence = top_prediction['confidence'] * 100

                if confidence >= confidence_threshold * 100:
                    st.success(f"**{uploaded_file.name}**: Тип почвы: **{soil_type}** (уверенность: {confidence:.2f}%)")
                    results.append([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        uploaded_file.name,
                        soil_type,
                        f"{confidence:.2f}"
                    ])
                else:
                    st.warning(f"**{uploaded_file.name}**: Уверенность ({confidence:.2f}%) ниже порога ({confidence_threshold * 100}%).")
                    logger.warning(f"Low confidence for {uploaded_file.name}: {confidence:.2f}%")
            else:
                st.error(f"**{uploaded_file.name}**: Не удалось определить тип почвы.")
                logger.error(f"No predictions for {uploaded_file.name}")

        except Exception as e:
            st.error(f"Ошибка при обработке {uploaded_file.name}: {str(e)}")
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if results:
        save_results_to_csv(results)
        df = pd.DataFrame(results, columns=["Timestamp", "Image Name", "Soil Type", "Confidence"])
        st.subheader("Таблица результатов")
        st.dataframe(df)
        st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

        st.subheader("Статистика уверенности")
        plot_confidence_histogram(results)

st.subheader("О модели классификации почвы")
st.markdown("""
    ### Информация о модели
    Приложение использует модель из **Roboflow Universe** (ID: `soil-type-model/1`), которая основана на архитектуре глубокого обучения (обычно CNN, например, ResNet или EfficientNet). Модель обучена на датасете изображений почвы, размеченных по типам (глина, песок, суглинок и т.д.).

    #### Как работает модель:
    - **Архитектура**: Конволюционная нейронная сеть (CNN) извлекает признаки из изображений (текстуру, цвет, зернистость).
    - **Обучение**: Модель обучена на тысячах изображений почвы, где каждое фото размечено по классу (например, clay, sand, loam). Датасет включает разнообразные условия освещения и текстуры.
    - **Классификация**: На вход подаётся изображение, модель возвращает вероятности для каждого класса почвы. Класс с наивысшей вероятностью выбирается как результат.
    - **Точность**: Зависит от качества датасета и условий съёмки. Для точных результатов используйте чёткие фото крупным планом.

    #### Ограничения:
    - **Качество изображения**: Размытые или плохо освещённые фото снижают точность.
    - **Ограниченные классы**: Модель распознаёт только те типы почвы, на которых обучена (например, не различает редкие подтипы).

    Для получения API-ключа и доступа к модели зарегистрируйтесь на https://app.roboflow.com.
""")