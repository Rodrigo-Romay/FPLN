import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense

# Funciones auxiliares
def generar_muestras_cbow(secuencia, window_size=2):
    entradas_contexto = []
    salidas_objetivo = []
    for i in range(window_size, len(secuencia) - window_size):
        contexto_izq = secuencia[i - window_size : i]
        contexto_der = secuencia[i + 1 : i + window_size + 1]
        contexto = contexto_izq + contexto_der
        entradas_contexto.append(contexto)
        salidas_objetivo.append(secuencia[i])
    return np.array(entradas_contexto), np.array(salidas_objetivo)

def obtener_palabras_similares(palabra_objetivo, matriz_embeddings, word_index, index_word, top_n=10):
    if palabra_objetivo not in word_index:
        print(f"La palabra '{palabra_objetivo}' no está en el vocabulario.")
        return
    word_id = word_index[palabra_objetivo]
    word_vector = matriz_embeddings[word_id].reshape(1, -1)
    similitudes = cosine_similarity(word_vector, matriz_embeddings)[0]
    indices_top = np.argsort(similitudes)[::-1][1:top_n+1]
    
    print(f"\nLas {top_n} palabras más similares a '{palabra_objetivo}':")
    for idx in indices_top:
        if idx in index_word:
            print(f" - {index_word[idx]} (Similitud: {similitudes[idx]:.4f})")

def visualize_tsne_embeddings(words, embeddings, word_index, titulo="t-SNE CBOW", filename="tsne_cbow.png"):
    words_present = [w for w in words if w in word_index]
    if not words_present:
        return
        
    indices = [word_index[word] for word in words_present]
    selected_embeddings = embeddings[indices]
    perplexity = min(5, len(words_present) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words_present):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker='o')
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                    xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title(titulo)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico guardado: {filename}")

def visualize_all_tsne_embeddings(embeddings, word_index, words_to_plot, words_to_label=None, filename="tsne_all_cbow.png"):
    index_word = {index: word for word, index in word_index.items()}
    if words_to_label is None:
        words_to_label = words_to_plot
    words_to_label = set(words_to_label).intersection(words_to_plot)

    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = embeddings[indices_to_plot]
    perplexity = min(5, len(words_to_plot) - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(12, 12))
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:
            plt.annotate(index_word[index], xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                        xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Visualización t-SNE de todos los Embeddings (CBOW)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico guardado: {filename}")

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

if __name__ == '__main__':
    # Lectura y tokenización
    ruta_archivo = os.path.join('datasets', 'game_of_thrones.txt') 
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_corpus = f.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([texto_corpus])
    secuencia_ids = tokenizer.texts_to_sequences([texto_corpus])[0]
    vocab_size = len(tokenizer.word_index) + 1 

    print(f"Tamaño del vocabulario: {vocab_size}")

    # Creación de muestras
    print("\n Generando Muestras")
    window_size = 2
    X_cbow, y_cbow = generar_muestras_cbow(secuencia_ids, window_size=window_size)

    # Arquitectura del modelo
    print("\n Construyendo Modelo CBOW")
    embedding_size = 64
    context_length = window_size * 2

    entrada = Input(shape=(context_length,), name='entrada_contexto')
    capa_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, name='capa_embedding')(entrada)
    media_vectores = Lambda(lambda x: K.mean(x, axis=1), name='media_vectores')(capa_embedding)
    salida = Dense(vocab_size, activation='softmax', name='salida_palabra')(media_vectores)

    modelo_cbow = Model(inputs=entrada, outputs=salida)
    modelo_cbow.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EXTRACCIÓN Y VISUALIZACIÓN ANTES DEL ENTRENAMIENTO
    print("\n Extrayendo embeddings aleatorios antes del entrenamiento")
    pesos_antes = modelo_cbow.get_layer('capa_embedding').get_weights()[0]
    
    ruta_target = os.path.join('mats', 'target_words_game_of_thrones.txt')
    try:
        with open(ruta_target, 'r', encoding='utf-8') as f:
            palabras_objetivo = [linea.strip().lower() for linea in f.readlines() if linea.strip()]
        
        visualize_tsne_embeddings(palabras_objetivo, pesos_antes, tokenizer.word_index, 
                                titulo="t-SNE CBOW - ANTES de Entrenar", 
                                filename="tsne_cbow_antes.png")
    except FileNotFoundError:
        print(f"No se ha encontrado el archivo de palabras objetivo en: {ruta_target}")


    print("\n Entrenamiento del modelo")
    history = modelo_cbow.fit(X_cbow, y_cbow, batch_size=256, epochs=10, validation_split=0.1)
    plot_history(history, model_name="CBOW")

    # EVALUACIÓN CUANTITATIVA
    print("\n Evaluación Cuantitativa (después del entrenamiento)")
    pesos_despues = modelo_cbow.get_layer('capa_embedding').get_weights()[0]
    
    std_dev = np.std(pesos_despues)
    print(f"Desviación típica de los embeddings: {std_dev:.4f}")
    if std_dev < 0.05:
        print("Los embeddings podrían estar degenerados.")
    else:
        print("Embeddings saludables.")

    # EVALUACIÓN CUALITATIVA (Coseno y t-SNE)
    print("\n Evaluación Cualitativa")
    index_word = {id: palabra for palabra, id in tokenizer.word_index.items()}
    
    for palabra in palabras_objetivo:
        obtener_palabras_similares(palabra, pesos_despues, tokenizer.word_index, index_word, top_n=10)

    try:
        # Generar t-SNE (solo palabras objetivo) DESPUÉS de entrenar
        visualize_tsne_embeddings(palabras_objetivo, pesos_despues, tokenizer.word_index, 
                                titulo="t-SNE CBOW - DESPUÉS de Entrenar", 
                                filename="tsne_cbow_despues.png")

        # Generar t-SNE con TODAS las palabras de fondo
        palabras_fondo = list(tokenizer.word_index.keys())[:500] 
        # Añadimos nuestras palabras objetivo para asegurarnos de que estén en la gráfica
        palabras_fondo.extend(palabras_objetivo) 
        
        visualize_all_tsne_embeddings(pesos_despues, tokenizer.word_index, 
                                    words_to_plot=palabras_fondo, 
                                    words_to_label=palabras_objetivo, 
                                    filename="tsne_all_cbow_despues.png")
    except NameError:
        pass