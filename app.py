import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import leafmap.foliumap as leafmap

# Diretórios para armazenar imagens rotuladas
os.makedirs("data/positive", exist_ok=True)
os.makedirs("data/negative", exist_ok=True)

st.title("Classificador de Piscinas em Imagens")
st.write("Faça upload de imagens de satélite e rotule como contendo ou não piscina.")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Tem piscina"):
            image.save(f"data/positive/{uploaded_file.name}")
            st.success("Imagem salva como POSITIVA (com piscina)")

    with col2:
        if st.button("❌ Não tem piscina"):
            image.save(f"data/negative/{uploaded_file.name}")
            st.success("Imagem salva como NEGATIVA (sem piscina)")

st.markdown("---")
st.write("Total de imagens salvas:")
pos_count = len(os.listdir("data/positive"))
neg_count = len(os.listdir("data/negative"))
st.write(f"✅ Positivas: {pos_count}")
st.write(f"❌ Negativas: {neg_count}")

# Treinamento do modelo
if st.button("🚀 Treinar Modelo"):
    st.write("Preparando dados para treinamento...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        'data',
        target_size=(64, 64),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        'data',
        target_size=(64, 64),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    st.write("Construindo modelo CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    st.write("Treinando modelo...")
    history = model.fit(train_data, epochs=5, validation_data=val_data)
    st.success("Modelo treinado com sucesso!")

    model.save("pool_classifier_model.h5")
    st.write("Modelo salvo como pool_classifier_model.h5")

    # Plotar métricas
    st.write("### Desempenho do Treinamento")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Loss')
    ax2.legend()
    st.pyplot(fig)

# Mapa interativo com Leafmap
st.markdown("---")
st.subheader("🗺️ Mapa Interativo para Detecção de Piscinas")
st.write("Navegue pelo mapa abaixo e escolha uma área para detectar piscinas.")

with st.expander("Mostrar mapa interativo"):
    m = leafmap.Map(center=[-23.55, -46.63], zoom=16)
    m.add_basemap("OpenStreetMap")
    m.to_streamlit(height=500)

st.warning("⚠️ Integração automática com imagens do mapa ainda requer API externa ou captura manual. Para testes locais, baixe um screenshot do mapa e envie para o classificador acima.")
