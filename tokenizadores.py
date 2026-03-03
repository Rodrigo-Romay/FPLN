import regex as re
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import matplotlib.pyplot as plt

# ==========================================
# 1. Tokenización por espacios
# ==========================================
def tok_espacios(frase):
    return frase.split(" ")

# ==========================================
# 2. Tokenización por signos de puntuación
# ==========================================
def tokenizar_signos(texto):
    patron = r'\w+|[^\w\s]|\p{So}'
    tokens = re.findall(patron, texto, flags=re.UNICODE)
    return [t for t in tokens if t != '\u200d']

# ==========================================
# 3. Tokenización por n-gramas
# ==========================================
def tokenizar_ngramas(texto, n):
    if n < 1:
        raise ValueError("El valor de n debe ser mayor o igual a 1")
    
    tokens = tok_espacios(texto)
    ngramas = []
    
    for i in range(len(tokens) - n + 1):
        ventana = tokens[i : i + n]
        ngramas.append(" ".join(ventana))
        
    return ngramas

# ==========================================
# 4. Tokenización por Clasificación Supervisada
# ==========================================
def extraer_caracteristicas(texto, i):
    char = texto[i]
    next_char = texto[i+1] if i + 1 < len(texto) else "" 
    
    is_numeric = char.isnumeric()
    is_punct = not char.isalnum() and not char.isspace() 
    
    return {
        'char': char,
        'next_char': next_char,
        'is_numeric': is_numeric,
        'is_punct': is_punct
    }

def preparar_datos_entrenamiento(oraciones):
    X = []
    y = []
    
    patron = r'\w+|[^\w\s]|\p{So}'
    
    for oracion in oraciones:
        spans = [(m.start(), m.end()) for m in re.finditer(patron, oracion)]
        # El último carácter de cada token es la frontera (1)
        fronteras = set([end - 1 for start, end in spans])
        
        for i in range(len(oracion)):
            if oracion[i].isspace():
                continue
                
            X.append(extraer_caracteristicas(oracion, i))
            y.append(1 if i in fronteras else 0)
            
    return X, y

def entrenar_clasificador(oraciones_train):
    X_dicts, y = preparar_datos_entrenamiento(oraciones_train)
    
    vectorizador = DictVectorizer(sparse=False)
    X_vect = vectorizador.fit_transform(X_dicts)
    
    clasificador = LogisticRegression(max_iter=1000)
    clasificador.fit(X_vect, y)
    
    return clasificador, vectorizador

def tokenizar_clasificacion(texto, clasificador, vectorizador):
    tokens = []
    token_actual = ""
    
    for i in range(len(texto)):
        char = texto[i]
        
        if char.isspace():
            if token_actual:
                tokens.append(token_actual)
                token_actual = ""
            continue
            
        caracteristicas = extraer_caracteristicas(texto, i)
        x_vec = vectorizador.transform([caracteristicas])
        
        es_frontera = clasificador.predict(x_vec)[0]
        token_actual += char
        
        if es_frontera == 1:
            tokens.append(token_actual)
            token_actual = ""
            
    if token_actual:
        tokens.append(token_actual)
        
    return tokens

# ==========================================
# 5. Tokenización WordPiece
# ==========================================

def entrenar_wordpiece(oraciones_train, max_vocab):
    # Preprocesar corpus (tokenización por espacios) y Calcular frecuencias
    word_freqs = defaultdict(int)
    for oracion in oraciones_train:
        palabras = tok_espacios(oracion)
        for palabra in palabras:
            word_freqs[palabra] += 1

    vocab = set(["[UNK]"])
    
    # Segmentar en caracteres y añadir prefijos '##'
    word_splits = {}
    for word in word_freqs.keys():
        if not word: continue
        # El primer carácter va normal, los siguientes con '##'
        split = [word[0]] + ["##" + c for c in word[1:]]
        word_splits[word] = split
        # Añadimos todos los caracteres al vocabulario inicial
        for token in split:
            vocab.add(token)
            
    # Fusión Iterativa hasta alcanzar max_vocab
    while len(vocab) < max_vocab:
        # Calcular frecuencias de todos los pares adyacentes actuales
        pair_freqs = defaultdict(int)
        for word, split in word_splits.items():
            freq = word_freqs[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i+1])] += freq
                
        # Si no hay pares posibles, paramos
        if not pair_freqs:
            break
            
        # Encontrar el par más frecuente
        best_pair = max(pair_freqs, key=pair_freqs.get)
        left, right = best_pair
        
        # Crear la nueva unidad fusionada: si right tiene '##', se lo quitamos
        if right.startswith("##"):
            merged = left + right[2:]
        else:
            merged = left + right
            
        # Añadir el nuevo par fusionado al vocabulario
        vocab.add(merged)
        
        # Actualizar la segmentación de TODAS las palabras en el corpus
        for word, split in word_splits.items():
            if len(split) == 1:
                continue
                
            new_split = []
            i = 0
            while i < len(split):
                # Si encontramos el par, insertamos la versión fusionada y saltamos un índice
                if i < len(split) - 1 and split[i] == left and split[i+1] == right:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            word_splits[word] = new_split
            
    return vocab

def tokenizar_wordpiece(texto, vocab):
    # Segmentación inicial por espacios
    palabras = tok_espacios(texto)
    tokens_finales = []
    
    for palabra in palabras:
        if not palabra: 
            continue

        # Búsqueda del primer segmento válido desde el final hacia el principio
        primer_segmento = None

        # El tamaño del primer segmento debe ser mínimo de dos caracteres
        rango_minimo = 2 if len(palabra) >= 2 else 1
        
        # Se realiza eliminando caracteres desde el final hasta encontrar coincidencia
        for i in range(len(palabra), rango_minimo - 1, -1):
            if palabra[:i] in vocab:
                primer_segmento = palabra[:i]
                resto_idx = i
                break
                
        # Si no hay coincidencia para el primer segmento, se pone [UNK]
        if not primer_segmento:
            tokens_finales.append("[UNK]")
            continue
            
        tokens_palabra = [primer_segmento]
        idx = resto_idx
        palabra_es_desconocida = False
        
        # Tokenización del resto: incremental desde el principio hacia el final
        while idx < len(palabra):
            mejor_sub = None
            mejor_j = idx
            
            # Búsqueda incremental (de izquierda a derecha)
            for j in range(idx + 1, len(palabra) + 1):
                sub = "##" + palabra[idx:j]
                if sub in vocab:
                    # Nos quedamos con la coincidencia válida más larga encontrada en este barrido
                    mejor_sub = sub
                    mejor_j = j
                    
            if mejor_sub is not None:
                tokens_palabra.append(mejor_sub)
                idx = mejor_j
            else:
                # Si no encontramos ninguna subpalabra válida para continuar
                palabra_es_desconocida = True
                break
                
        if palabra_es_desconocida:
            tokens_finales.append("[UNK]")
        else:
            tokens_finales.extend(tokens_palabra)
            
    return tokens_finales

# ==========================================
# 6. Tokenización BPE
# ==========================================

def entrenar_bpe(oraciones_train, max_vocab):
    # Preprocesar y calcular frecuencias
    word_freqs = defaultdict(int)
    for oracion in oraciones_train:
        palabras = tok_espacios(oracion)
        for palabra in palabras:
            word_freqs[palabra] += 1
            
    # Segmentar en caracteres individuales y crear vocabulario inicial
    vocab = set()
    word_splits = {}
    for word in word_freqs.keys():
        if not word: continue
        split = list(word) # Dividir directamente en caracteres (sin prefijos)
        word_splits[word] = split
        for char in split:
            vocab.add(char)
            
    reglas_fusion = []
    
    # Fusión Iterativa hasta alcanzar max_vocab
    while len(vocab) < max_vocab:
        # Calcular frecuencias de todos los pares adyacentes actuales
        pair_freqs = defaultdict(int)
        for word, split in word_splits.items():
            freq = word_freqs[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i+1])] += freq
                
        if not pair_freqs:
            break # No hay más pares que fusionar
            
        # Encontrar el par más frecuente
        best_pair = max(pair_freqs, key=pair_freqs.get)
        left, right = best_pair
        merged = left + right
        
        # Añadir al vocabulario y guardar la regla de fusión
        vocab.add(merged)
        reglas_fusion.append((best_pair, merged))
        
        # Actualizar la segmentación de TODAS las palabras
        for word, split in word_splits.items():
            if len(split) == 1:
                continue
                
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == left and split[i+1] == right:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            word_splits[word] = new_split
            
    return vocab, reglas_fusion

def tokenizar_bpe(texto, reglas_fusion):
    # Segmentación inicial por espacios
    palabras = tok_espacios(texto)
    tokens_finales = []
    
    for palabra in palabras:
        if not palabra: continue
        
        # División en caracteres iniciales
        split = list(palabra)
        
        # Aplicación de reglas de fusión en orden
        for (left, right), merged in reglas_fusion:
            # Si el split ya es de 1 elemento, no hay nada que fusionar
            if len(split) < 2:
                continue
                
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == left and split[i+1] == right:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
            
        # Añadimos los tokens resultantes de esta palabra
        tokens_finales.extend(split)
        
    return tokens_finales

# ==========================================
# REPRESENTACIÓN GRÁFICA DE RESULTADOS
# ==========================================

def generar_grafica_vocabulario(path_speeches, oraciones_train):
    # Cargamos los ficheros de prueba
    try:
        with open(path_speeches, "r", encoding="utf-8") as f:
            discursos = [linea.rstrip('\n\r') for linea in f if linea.strip()]
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {path_speeches}")
        return

    # Entrenamos los modelos
    print("Entrenando modelos para la comparativa...")
    
    # Configuración del clasificador 
    clasificador, vectorizador = entrenar_clasificador(oraciones_train)
    
    # Ponemos el límite para Wordpiece y BPE de 3000 
    vocab_wp = entrenar_wordpiece(discursos, max_vocab=3000)
    _, reglas_bpe = entrenar_bpe(discursos, max_vocab=3000)

    metodos = {
        "Espacios": lambda x: tok_espacios(x),
        "Signos Puntuación": lambda x: tokenizar_signos(x),
        "N-gramas (n=2)": lambda x: tokenizar_ngramas(x, 2),
        "Clasif. Supervisado": lambda x: tokenizar_clasificacion(x, clasificador, vectorizador),
        "WordPiece (3000)": lambda x: tokenizar_wordpiece(x, vocab_wp),
        "BPE (3000)": lambda x: tokenizar_bpe(x, reglas_bpe)
    }

    vocabularios_vistos = {nombre: set() for nombre in metodos}
    evolucion = {nombre: [] for nombre in metodos}
    eje_x = []

    # Procesamos las oraciones y almacenamos la evolución
    print("Procesando oraciones para la gráfica...")
    for i, oracion in enumerate(discursos):
        for nombre, func_tok in metodos.items():
            tokens = func_tok(oracion)
            vocabularios_vistos[nombre].update(tokens)
            evolucion[nombre].append(len(vocabularios_vistos[nombre]))
        
        eje_x.append(i + 1)

    plt.figure(figsize=(12, 7))
    for nombre in metodos:
        plt.plot(eje_x, evolucion[nombre], label=nombre)

    plt.title("Evolución del tamaño del vocabulario por método de tokenización")
    plt.xlabel("Número de oraciones procesadas")
    plt.ylabel("Número de tokens únicos (Vocabulario)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar y mostramos la gráfica
    plt.savefig("evolucion_vocabulario.png")
    print("Gráfica guardada como 'evolucion_vocabulario.png'")
    plt.show()


# ==========================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    # Cargar los datos
    oraciones_test = []
    oraciones_train = []
    
    with open("test_sentences.txt", "r", encoding="utf-8") as f:
        oraciones_test = [linea.rstrip('\n\r') for linea in f if linea.strip()]
        
    with open("training_sentences.txt", "r", encoding="utf-8") as f:
        oraciones_train = [linea.rstrip('\n\r') for linea in f if linea.strip()]

    clasificador_rl, vectorizador_rl = entrenar_clasificador(oraciones_train)

    # Evaluación de resultados
    print("\n=== Tokenización por espacios (Test) ===")
    for oracion in oraciones_test: 
        print(f"Input: '{oracion}' -> Tokens: {tok_espacios(oracion)}")
        
    print("\n=== Tokenización por espacios (Train) ===")
    for oracion in oraciones_train: 
        print(f"Input: '{oracion}' -> Tokens: {tok_espacios(oracion)}")

    print("\n=== Tokenización por signos de puntuación (Test) ===")
    for oracion in oraciones_test:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_signos(oracion)}")

    print("\n=== Tokenización por signos de puntuación (Train) ===")
    for oracion in oraciones_train:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_signos(oracion)}")
        
    print("\n=== Tokenización por n-gramas (n=2) (Test) ===")
    for oracion in oraciones_test:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_ngramas(oracion, 2)}")

    print("\n=== Tokenización por n-gramas (n=2) (Train) ===")
    for oracion in oraciones_train:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_ngramas(oracion, 2)}")

    print("\n=== Tokenización por Clasificación Supervisada (Test)===")
    for oracion in oraciones_test:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_clasificacion(oracion, clasificador_rl, vectorizador_rl)}")

    print("\n=== Tokenización por Clasificación Supervisada (Train) ===")
    for oracion in oraciones_train:
        print(f"Input: '{oracion}' -> Tokens: {tokenizar_clasificacion(oracion, clasificador_rl, vectorizador_rl)}")
        
        
    print("\n=== Tokenización WordPiece (Subpalabras) ===")
    tamaños_vocabulario = [100, 150, 200]
    for tamaño in tamaños_vocabulario:
        print(f"\n--- Entrenando WordPiece (Vocabulario: {tamaño}) ---")
        vocab_wp = entrenar_wordpiece(oraciones_train, tamaño)
        print(f"Vocabulario resultante: {sorted(list(vocab_wp))}\n")
        
        print("\n=== Tokenización WordPiece (Test)===")

        for oracion in oraciones_test:
            tokens_wp = tokenizar_wordpiece(oracion, vocab_wp)
            print(f"Input: '{oracion}' -> Tokens: {tokens_wp}")

        print("\n=== Tokenización WordPiece (Train)===")
        for oracion in oraciones_train:
            tokens_wp = tokenizar_wordpiece(oracion, vocab_wp)
            print(f"Input: '{oracion}' -> Tokens: {tokens_wp}")

    print("\n=== Tokenización BPE (Emparejamiento de pares) ===")
    for tamaño in tamaños_vocabulario:
        print(f"\n--- Entrenando BPE (Vocabulario: {tamaño}) ---")
        vocab_bpe, reglas_bpe = entrenar_bpe(oraciones_train, tamaño)
        print(f"Vocabulario resultante: {sorted(list(vocab_bpe))}\n")
        
        print("\n=== Tokenización BPE (Test)===")
        for oracion in oraciones_test:
            tokens_bpe = tokenizar_bpe(oracion, reglas_bpe)
            print(f"Input: '{oracion}' -> Tokens: {tokens_bpe}")

        print("\n=== Tokenización BPE (Train)===")
        for oracion in oraciones_train:
            tokens_bpe = tokenizar_bpe(oracion, reglas_bpe)
            print(f"Input: '{oracion}' -> Tokens: {tokens_bpe}")
            
    print("\nAnálisis de evolución del vocabulario...")
    
    generar_grafica_vocabulario("majesty_speeches.txt", oraciones_train)
    
    print("\nProceso finalizado. ")


