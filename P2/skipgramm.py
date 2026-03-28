import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import skipgrams
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def cargar_y_tokenizar(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_corpus = f.read()

    print("Entrenando el Tokenizador...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([texto_corpus])

    secuencia_ids = tokenizer.texts_to_sequences([texto_corpus])[0]
    vocab_size = len(tokenizer.word_index) + 1

    print(f"Tamaño del vocabulario: {vocab_size}")
    print(f"Total de palabras en el texto: {len(secuencia_ids)}")

    return tokenizer, secuencia_ids, vocab_size


def submuestreo(secuencia, t=1e-3):
    """
    Descarta palabras muy frecuentes con probabilidad P(w) = 1 - sqrt(t / f(w))
    donde f(w) es la frecuencia relativa de la palabra en el corpus.
    Fórmula de Mikolov et al. (2013), umbral t=1e-3 por defecto.
    """
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


def generar_pares(secuencia_submuestreada, vocab_size, window_size=6, negative_samples=4.0):
    pairs, labels = skipgrams(
        secuencia_submuestreada,
        vocabulary_size=vocab_size,
        window_size=window_size,
        negative_samples=negative_samples,
        seed=42
    )

    word_target, word_context = zip(*pairs)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    labels = np.array(labels, dtype="int32")

    print(f"Parejas generadas tras submuestreo: {len(pairs)}")
    return word_target, word_context, labels


def construir_modelo(vocab_size, embedding_dim=100):
    input_target = Input((1,), name="entrada_palabra_central")
    input_context = Input((1,), name="entrada_palabra_contexto")

    embedding_layer = Embedding(vocab_size, embedding_dim, name="capa_embedding")

    target_vector = embedding_layer(input_target)
    context_vector = embedding_layer(input_context)

    dot_product = Dot(axes=2)([target_vector, context_vector])
    dot_product = Flatten()(dot_product)

    output = Dense(1, activation="sigmoid")(dot_product)

    modelo = Model(inputs=[input_target, input_context], outputs=output)
    modelo.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    modelo.summary()

    return modelo


def entrenar_modelo(modelo, word_target, word_context, labels, batch_size=1024, epochs=8):
    print("\nIniciando el entrenamiento...")

    historial = modelo.fit(
        x=[word_target, word_context],
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1
    )

    pesos_embeddings = modelo.get_layer('capa_embedding').get_weights()[0]
    print(f"\nForma final de la matriz de embeddings: {pesos_embeddings.shape}")

    return historial, pesos_embeddings


def obtener_palabras_similares(palabra_objetivo, matriz_embeddings, tokenizer, top_n=10):
    index_word = {id: palabra for palabra, id in tokenizer.word_index.items()}

    if palabra_objetivo not in tokenizer.word_index:
        print(f"La palabra '{palabra_objetivo}' no está en el vocabulario.")
        return

    word_id = tokenizer.word_index[palabra_objetivo]
    word_vector = matriz_embeddings[word_id].reshape(1, -1)

    similitudes = cosine_similarity(word_vector, matriz_embeddings)[0]
    indices_top = np.argsort(similitudes)[::-1][1:top_n + 1]

    print(f"\nLas {top_n} palabras más similares a '{palabra_objetivo}':")
    for idx in indices_top:
        if idx in index_word:
            print(f" - {index_word[idx]} (Similitud: {similitudes[idx]:.4f})")


def evaluar_similitudes(pesos_embeddings, tokenizer, palabras_prueba):
    print("\n--- Evaluando Similitud Semántica ---")
    for palabra in palabras_prueba:
        obtener_palabras_similares(palabra, pesos_embeddings, tokenizer, top_n=10)


def comprobar_embeddings(pesos_embeddings):
    print("\n--- Comprobación de Embeddings Degenerados ---")
    std_dev = np.std(pesos_embeddings)
    print(f"Desviación típica de los embeddings: {std_dev:.4f}")

    if std_dev < 0.05:
        print("[FAIL] Cuidado: Los embeddings podrían estar degenerados (colapsados en la misma región).")
    else:
        print("[OK] Embeddings saludables. Existe una dispersión correcta en el espacio vectorial.")


def visualize_tsne_embeddings(words, embeddings, word_index, filename=None):
    print("\n--- Generando Visualización t-SNE ---")
    pesos_normalizados = normalize(embeddings)
    words_present = [w for w in words if w in word_index]

    if not words_present:
        print("Ninguna de las palabras objetivo está en el vocabulario.")
        return

    indices = [word_index[word] for word in words_present]
    selected_embeddings = pesos_normalizados[indices]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=15,
        learning_rate='auto',
        init='pca',
        n_iter=2000,
        random_state=42
    )

    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words_present):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker='o')
        plt.annotate(word,
                     xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom')

    plt.title("Visualización t-SNE de Word Embeddings (Skipgram)")
    plt.grid(True, linestyle='--', alpha=0.6)

    if filename:
        plt.savefig(filename)
        print(f"Gráfico guardado como: {filename}")
    else:
        plt.show()


def cargar_palabras_objetivo(ruta_target):
    try:
        with open(ruta_target, 'r', encoding='utf-8') as f:
            palabras = [linea.strip().lower() for linea in f.readlines() if linea.strip()]
        print(f"Se han cargado {len(palabras)} palabras para visualizar.")
        return palabras
    except FileNotFoundError:
        print(f"No se ha encontrado el archivo en la ruta: {ruta_target}")
        return []


if __name__ == "__main__":
    # Rutas
    ruta_archivo = os.path.join('datasets', 'game_of_thrones.txt')
    ruta_target  = os.path.join('mats', 'target_words_game_of_thrones.txt')

    # Pipeline
    tokenizer, secuencia_ids, vocab_size = cargar_y_tokenizar(ruta_archivo)

    print("Aplicando submuestreo para limpiar el texto...")
    secuencia_submuestreada = submuestreo(secuencia_ids, t=1e-3)
    print(f"Palabras originales: {len(secuencia_ids)}")
    print(f"Palabras tras el filtro: {len(secuencia_submuestreada)}")

    word_target, word_context, labels = generar_pares(secuencia_submuestreada, vocab_size)

    modelo = construir_modelo(vocab_size, embedding_dim=100)

    historial, pesos_embeddings = entrenar_modelo(modelo, word_target, word_context, labels)

    evaluar_similitudes(pesos_embeddings, tokenizer, palabras_prueba=["betray", "ambitious", "sword", "wall"])

    comprobar_embeddings(pesos_embeddings)

    palabras_objetivo = cargar_palabras_objetivo(ruta_target)
    if palabras_objetivo:
        visualize_tsne_embeddings(palabras_objetivo, pesos_embeddings, tokenizer.word_index, filename="tsne_skipgram.png")