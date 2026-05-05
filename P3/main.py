from materiales.conllu_reader import ConlluReader
from materiales.algorithm import ArcEager
from materiales.model import ParserMLP
from materiales.postprocessor import PostProcessor

def cargar_y_preparar_datos(reader):
    """Carga los ficheros CoNLL-U y filtra oraciones no proyectivas[cite: 1, 7]."""
    print("--- 1. Cargando y filtrando datos ---")
    
    train_trees = reader.read_conllu_file("materiales/en_partut-ud-train_clean.conllu")
    dev_trees = reader.read_conllu_file("materiales/en_partut-ud-dev_clean.conllu")
    test_trees = reader.read_conllu_file("materiales/en_partut-ud-test_clean.conllu")
    
    # El algoritmo Arc-Eager requiere árboles proyectivos en entrenamiento y desarrollo[cite: 1, 2].
    train_proj = reader.remove_non_projective_trees(train_trees)
    dev_proj = reader.remove_non_projective_trees(dev_trees)
    
    print(f"Entrenamiento: {len(train_proj)} oraciones proyectivas (de {len(train_trees)}).")
    print(f"Desarrollo: {len(dev_proj)} oraciones proyectivas (de {len(dev_trees)}).")
    print(f"Test: {len(test_trees)} oraciones cargadas para inferencia.\n")
    
    return train_proj, dev_proj, test_trees

def generar_muestras_oraculo(trees, arc_eager, tipo="entrenamiento"):
    """Transforma árboles sintácticos en muestras de (estado, transición)[cite: 1, 7]."""
    print(f"--- 2. Generando muestras de {tipo} (Oracle) ---")
    samples = []
    for tree in trees:
        samples.extend(arc_eager.oracle(tree))
    print(f"Total de muestras generadas para {tipo}: {len(samples)}.\n")
    return samples

def entrenar_analizador(train_samples, dev_samples):
    """Instancia y entrena el modelo MLP con validación por época."""
    print("--- 3. Entrenamiento del modelo neuronal ---")
    # Subimos a 30 épocas para mejorar el aprendizaje multitarea[cite: 8, 11].
    model = ParserMLP(
        word_emb_dim=100, 
        hidden_dim=256, 
        epochs=30, 
        batch_size=128
    )
    
    # El método train mostrará la pérdida y precisión de validación por cada época[cite: 8, 11].
    model.train(train_samples, dev_samples)
    print("Entrenamiento finalizado.\n")
    return model

def realizar_inferencia_y_postproceso(model, test_trees, reader):
    """Ejecuta el parser, guarda resultados y aplica correcciones estructurales[cite: 1, 9, 10]."""
    print("--- 4. Inferencia y Postprocesado ---")
    
    # Procesamiento eficiente "en vertical" del conjunto de test[cite: 1].
    parsed_trees = model.run(test_trees)
    
    fichero_raw = "materiales/predicciones_raw.conllu"
    reader.write_conllu_file(fichero_raw, parsed_trees)
    
    print("Aplicando PostProcessor para corregir árboles malformados...")
    postprocessor = PostProcessor()
    # Asegura que cada oración tenga una única raíz según el estándar UD[cite: 1, 10].
    final_trees = postprocessor.postprocess(fichero_raw)
    
    fichero_final = "materiales/predicciones_final.conllu"
    reader.write_conllu_file(fichero_final, final_trees)
    
    print(f"Resultados finales guardados en: {fichero_final}\n")

def main():
    """Función principal que orquestra todo el proceso de la Práctica 3."""
    reader = ConlluReader()
    arc_eager = ArcEager()
    
    # Paso 1: Datos
    train_proj, dev_proj, test_trees = cargar_y_preparar_datos(reader)
    
    # Paso 2: Muestras (Oracle)
    train_samples = generar_muestras_oraculo(train_proj, arc_eager, "entrenamiento")
    dev_samples = generar_muestras_oraculo(dev_proj, arc_eager, "desarrollo")
    
    # Paso 3: Modelo
    model = entrenar_analizador(train_samples, dev_samples)
    
    # Paso 4: Inferencia y Postprocesado
    realizar_inferencia_y_postproceso(model, test_trees, reader)
    
    print("--- Proceso completado con éxito ---")
    print("Siguiente paso: Ejecutar script de evaluación oficial (conll18_ud_eval.py).[cite: 1, 6]")

if __name__ == "__main__":
    main()