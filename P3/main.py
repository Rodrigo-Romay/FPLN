from materiales.conllu_reader import ConlluReader
from materiales.algorithm import ArcEager
from materiales.model import ParserMLP
from materiales.postprocessor import PostProcessor



def read_file(reader, path, inference = False):
    trees = reader.read_conllu_file(path, inference=inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="materiales/en_partut-ud-train_clean.conllu")
dev_trees = read_file(reader,path="materiales/en_partut-ud-dev_clean.conllu")
test_trees = read_file(reader,path="materiales/en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()

print ("\n ------ TODO: Implement the rest of the assignment ------")

# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.

print("Generando muestras de entrenamiento...")
train_samples = []
for tree in train_trees:
    train_samples.extend(arc_eager.oracle(tree))

print(f"Total de muestras de entrenamiento generadas: {len(train_samples)}")

print("Generando muestras de desarrollo (validación)...")
dev_samples = []
for tree in dev_trees:
    dev_samples.extend(arc_eager.oracle(tree))

print(f"Total de muestras de desarrollo generadas: {len(dev_samples)}")

# --- ENTRENAMIENTO ---
print("\n--- INICIANDO EL MODELO NEURONAL ---")
parser_model = ParserMLP(word_emb_dim=100, hidden_dim=200, epochs=20, batch_size=64)
parser_model.train(train_samples, dev_samples)

parser_model.evaluate(dev_samples)

print("\n Entrenamiento y evaluación terminados")

# --- INFERENCIA Y POSTPROCESADO ---
print("\n--- PREDICCIÓN EN CONJUNTO DE TEST ---")

# Predecimos la estructura de las oraciones de test
parser_model.run(test_trees)

# Guardamos el archivo con las predicciones sin arreglar
out_path_raw = "test_predictions_raw.conllu"
reader.write_conllu_file(out_path_raw, test_trees)
print(f"Predicciones guardadas en {out_path_raw}")

# Postprocesamos para arreglar árboles corruptos
print("\n--- POSTPROCESAMIENTO ---")
pp = PostProcessor()
fixed_trees = pp.postprocess(out_path_raw)

# Guardamos el archivo final y limpio
out_path_final = "test_predictions_final.conllu"
reader.write_conllu_file(out_path_final, fixed_trees)
print(f"Predicciones limpias guardadas en {out_path_final}")
print("¡Práctica completada con éxito!")