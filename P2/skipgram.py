import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import skipgrams

# Preprocesado
def cargar_y_tokenizar(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_corpus = f.read()

    print("Entrenando el Tokenizador...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([texto_corpus])
    secuencia_ids = tokenizer.texts_to_sequences([texto_corpus])[0]
    vocab_size = len(tokenizer.word_index) + 1

    print(f"Tamaño del vocabulario: {vocab_size}")
    return tokenizer, secuencia_ids, vocab_size

def generar_muestras_skipgram(secuencia, vocab_size, window_size=2, negative_samples=4):
    target_words = []
    context_words = []
    labels = []

    for i, target in enumerate(secuencia):
        # Definimos los límites de la ventana para la palabra actual
        inicio = max(0, i - window_size)
        fin = min(len(secuencia), i + window_size + 1)
        
        for j in range(inicio, fin):
            if i == j: continue  # No emparejar la palabra consigo misma
            
            # 1. PAR POSITIVO (Palabra central + Palabra en ventana)
            target_words.append(target)
            context_words.append(secuencia[j])
            labels.append(1)
            
            # 2. PARES NEGATIVOS (Muestreo Negativo)
            for _ in range(negative_samples):
                # Elegimos una palabra aleatoria del vocabulario
                negativo = np.random.randint(1, vocab_size)
                # Evitamos que la palabra negativa sea la propia palabra central
                while negativo == target:
                    negativo = np.random.randint(1, vocab_size)
                
                target_words.append(target)
                context_words.append(negativo)
                labels.append(0)

    return np.array(target_words), np.array(context_words), np.array(labels)

def submuestreo(secuencia, t=1e-5):
    total_words = len(secuencia)
    conteo = {}
    for word_id in secuencia:
        conteo[word_id] = conteo.get(word_id, 0) + 1

    secuencia_filtrada = []
    for word_id in secuencia:
        freq = conteo[word_id] / total_words
        prob_descartar = max(0.0, 1.0 - np.sqrt(t / freq))
        if np.random.random() > prob_descartar:
            secuencia_filtrada.append(word_id)
    return secuencia_filtrada

# Arquitectura del modelo

def construir_modelo(vocab_size, embedding_dim = 64):
    # Dos entradas independientes
    input_target = Input((1,), name="entrada_palabra_central")
    input_context = Input((1,), name="entrada_palabra_contexto")

    # Capa de embedding compartida
    embedding_layer = Embedding(vocab_size, embedding_dim, name="capa_embedding")

    target_vector = embedding_layer(input_target)
    context_vector = embedding_layer(input_context)

    # Producto escalar para medir afinidad
    dot_product = Dot(axes=2)([target_vector, context_vector])
    dot_product = Flatten()(dot_product)

    # Salida binaria
    output = Dense(1, activation="sigmoid")(dot_product)

    modelo = Model(inputs=[input_target, input_context], outputs=output)
    modelo.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return modelo

def obtener_palabras_similares(palabra_objetivo, matriz_embeddings, word_index, index_word, top_n=10):
    if palabra_objetivo not in word_index:
        print(f"La palabra '{palabra_objetivo}' no está en el vocabulario.")
        return
    
    word_id = word_index[palabra_objetivo]
    word_vector = matriz_embeddings[word_id].reshape(1, -1)
    
    # Calcular similitud de coseno contra todos los embeddings
    similitudes = cosine_similarity(word_vector, matriz_embeddings)[0]
    
    # Obtener los índices de los top_n más similares (excluyendo la propia palabra)
    indices_top = np.argsort(similitudes)[::-1][1:top_n+1]
    
    print(f"\nLas {top_n} palabras más similares a '{palabra_objetivo}':")
    for idx in indices_top:
        if idx in index_word:
            print(f" - {index_word[idx]} (Similitud: {similitudes[idx]:.4f})")

# VISUALIZACIÓN
def visualize_tsne_embeddings(words, embeddings, word_index, titulo, filename):
    words_present = [w for w in words if w in word_index]
    if not words_present: return
        
    indices = [word_index[word] for word in words_present]
    selected_embeddings = normalize(embeddings[indices])
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(words_present)-1), random_state=42)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words_present):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                    xytext=(5, 2), textcoords='offset points')
    plt.title(titulo)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()

def visualize_all_tsne_embeddings(embeddings, word_index, words_to_plot, words_to_label, filename):
    index_word = {index: word for word, index in word_index.items()}
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = normalize(embeddings[indices_to_plot])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(12, 12))
    words_to_label_set = set(words_to_label)
    
    for i, idx in enumerate(indices_to_plot):
        word = index_word[idx]
        alpha = 1.0 if word in words_to_label_set else 0.2
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=alpha)
        if word in words_to_label_set:
            plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), xytext=(5, 2), textcoords='offset points')
            
    plt.title("t-SNE Skip-gram: Palabras objetivo sobre el vocabulario")
    plt.savefig(filename)
    plt.close()

def plot_history(history, model_name="Modelo"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Métricas de Entrenamiento - {model_name}', fontsize=14, fontweight='bold')

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Entrenamiento', marker='o', color='tab:blue')
    ax1.plot(history.history['val_accuracy'], label='Validación', marker='o', color='tab:orange')
    ax1.set_title('Precisión (Accuracy)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Loss
    ax2.plot(history.history['loss'], label='Entrenamiento', marker='o', color='tab:blue')
    ax2.plot(history.history['val_loss'], label='Validación', marker='o', color='tab:orange')
    ax2.set_title('Pérdida (Loss)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    
    filename = f"metrics_{model_name.lower()}.png"
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Seteamos semillas para reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)

    ruta_archivo = os.path.join('datasets', 'game_of_thrones.txt')
    ruta_target  = os.path.join('mats', 'target_words_game_of_thrones.txt')

    tokenizer, secuencia_ids, vocab_size = cargar_y_tokenizar(ruta_archivo)
    secuencia_filtrada = submuestreo(secuencia_ids)

    # Generamos pares (1 positivo y 4 negativos)
    word_target, word_context, etiquetas = generar_muestras_skipgram(
        secuencia_filtrada, 
        vocab_size=vocab_size, 
        window_size=2, 
        negative_samples=4
    )

    # Cargamos palabras objetivo
    with open(ruta_target, 'r', encoding='utf-8') as f:
        palabras_objetivo = [l.strip().lower() for l in f.readlines() if l.strip()]

    # Modelo y Pesos ANTES
    modelo = construir_modelo(vocab_size)
    pesos_antes = modelo.get_layer('capa_embedding').get_weights()[0]
    visualize_tsne_embeddings(palabras_objetivo, pesos_antes, tokenizer.word_index, 
                            "t-SNE Skip-gram (ANTES)", "tsne_skipgram_antes.png")

    # ENTRENAMIENTO BALANCEADO
    # Como hay 4 negativos por cada 1 positivo, damos 4 veces más importancia al positivo
    print("\nIniciando entrenamiento balanceado...")
    pesos_clase = {0: 1.0, 1: 4.0} 
    
    history = modelo.fit(x=[word_target, word_context], y=etiquetas, 
                    batch_size=1024, epochs=5, validation_split=0.1, 
                    class_weight=pesos_clase)
    
    plot_history(history, model_name="SkipGram")

    # Evaluación y Pesos DESPUÉS
    pesos_despues = modelo.get_layer('capa_embedding').get_weights()[0]
    
    # Comprobación de salud
    std_dev = np.std(pesos_despues)
    print(f"\nDesviación típica: {std_dev:.4f} -> {'[OK]' if std_dev > 0.05 else '[FAIL]'}")

    # Visualizaciones DESPUÉS
    visualize_tsne_embeddings(palabras_objetivo, pesos_despues, tokenizer.word_index, 
                            "t-SNE Skip-gram (DESPUÉS)", "tsne_skipgram_despues.png")
    
    palabras_fondo = list(tokenizer.word_index.keys())[:500]
    visualize_all_tsne_embeddings(pesos_despues, tokenizer.word_index, 
                                palabras_fondo + palabras_objetivo, 
                                palabras_objetivo, "tsne_all_skipgram_despues.png")

    # Similitud
    print("\nEvaluación Cualitativa: Palabras Similares")
    index_word = {id: palabra for palabra, id in tokenizer.word_index.items()}
    
    for palabra in palabras_objetivo:
        obtener_palabras_similares(palabra, pesos_despues, tokenizer.word_index, index_word, top_n=10)