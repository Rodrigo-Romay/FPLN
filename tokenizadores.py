import regex as re
import sys

def tok_espacios(frase):
    return frase.split(" ")

def tokenizar_signos(texto):
    patron = r'\w+|[^\w\s]|\p{So}'
    tokens = re.findall(patron, texto, flags=re.UNICODE)
    return [t for t in tokens if t != '\u200d']
    
    return tokens

def tokenizar_ngramas(texto, n):
    if n < 1:
        raise ValueError("El valor de n debe ser >= 1")
    tokens = texto.split(" ")
    ngramas = []
    for i in range(len(tokens) - n + 1):
        ventana = tokens[i: i + n]
        ngramas.append(" ".join(ventana))
    return ngramas


with open("test_sentences.txt", encoding="utf-8") as f:
    oraciones = [linea.strip() for linea in f if linea.strip()]

print("=== Tokenización por espacios ===")
for oracion in oraciones:
    print(f"Input: '{oracion}' -> Tokens: {tok_espacios(oracion)}")

print("\n=== Tokenización por signos ===")
for oracion in oraciones:
    print(f"Input: '{oracion}' -> Tokens: {tokenizar_signos(oracion)}")
    
print("\n=== Tokenización por n-gramas ===")
for oracion in oraciones:
    print(f"Input: '{oracion}' -> Tokens: {tokenizar_ngramas(oracion, 2)}")
