import regex as re

def tok_espacios(frase):
    return frase.split(" ")

def tokenizar_signos(texto):
    patron = r'\w+|[^\w\s]|\p{So}'
    tokens = re.findall(patron, texto, flags = re.UNICODE)
    
    return tokens

def tokenizar_ngramas(texto, n):
    if n < 1:
        raise ValueError("El valor de n debe ser mayor o igual a 1")
    
    tokens = texto.split(" ")
    ngramas = []
    
    # El rango llega hasta len(tokens) - n + 1 para no salirnos del índice
    for i in range(len(tokens) - n + 1):
        ventana = tokens[i : i + n]
        ngramas.append(" ".join(ventana))
        
    return ngramas


print(tok_espacios('Hola me llamo juanjo.'))
print(tokenizar_signos("¡Hola me llamo Juanjo. 🤓😊"))
print(tokenizar_ngramas("Hola me llamo Juanjo. 🤓😊", 5))