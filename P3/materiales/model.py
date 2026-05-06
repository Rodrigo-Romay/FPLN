from .conllu_token import Token
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                epochs: int = 1, batch_size: int = 64):
        
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Diccionarios para convertir de String a ID
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.upos_to_id = {"<PAD>": 0, "<UNK>": 1}
        
        # Diccionarios para convertir de String a ID para la salida
        self.action_to_id = {}
        self.deprel_to_id = {}
        
        # Diccionarios inversos para hacer predicciones (ID a String)
        self.id_to_action = {}
        self.id_to_deprel = {}

        # El modelo de Keras que construiremos
        self.model = None
    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        
        # --- 1. CONSTRUIR DICCIONARIOS (VOCABULARIO) ---
        print("Construyendo vocabularios...")
        for sample in training_samples:
            feats = sample.state_to_feats()
            # Las primeras 4 features son palabras, las últimas 4 son UPOS
            for word in feats[:4]:
                if word not in self.word_to_id:
                    self.word_to_id[word] = len(self.word_to_id)
            for pos in feats[4:]:
                if pos not in self.upos_to_id:
                    self.upos_to_id[pos] = len(self.upos_to_id)
            
            # Registrar acciones (SHIFT, REDUCE, LA, RA)
            action = sample.transition.action
            if action not in self.action_to_id:
                idx = len(self.action_to_id)
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action
                
            # Registrar dependencias (solo si existen)
            dep = sample.transition.dependency
            if dep is not None and dep not in self.deprel_to_id:
                idx = len(self.deprel_to_id)
                self.deprel_to_id[dep] = idx
                self.id_to_deprel[idx] = dep

        # Añadimos una dependencia nula para cuando la acción es SHIFT o REDUCE
        if "None" not in self.deprel_to_id:
            idx = len(self.deprel_to_id)
            self.deprel_to_id["None"] = idx
            self.id_to_deprel[idx] = "None"

        print(f"Tamaño vocabulario palabras: {len(self.word_to_id)}")
        print(f"Tamaño vocabulario UPOS: {len(self.upos_to_id)}")
        print(f"Total acciones: {len(self.action_to_id)}")
        print(f"Total dependencias: {len(self.deprel_to_id)}")

        # --- 2. PREPARAR DATOS EN FORMATO NUMPY ---
        def extract_tensors(samples):
            X_words = np.zeros((len(samples), 4), dtype='int32')
            X_upos = np.zeros((len(samples), 4), dtype='int32')
            Y_action = np.zeros((len(samples),), dtype='int32')
            Y_deprel = np.zeros((len(samples),), dtype='int32')
            
            for i, s in enumerate(samples):
                feats = s.state_to_feats()
                # Extraemos y convertimos a IDs (usando <UNK> = 1 si no existe)
                X_words[i] = [self.word_to_id.get(w, 1) for w in feats[:4]]
                X_upos[i] = [self.upos_to_id.get(p, 1) for p in feats[4:]]
                
                Y_action[i] = self.action_to_id[s.transition.action]
                
                dep = s.transition.dependency if s.transition.dependency else "None"
                Y_deprel[i] = self.deprel_to_id.get(dep, self.deprel_to_id["None"])
                
            return [X_words, X_upos], [Y_action, Y_deprel]

        print("Extrayendo tensores de entrenamiento y desarrollo...")
        X_train, Y_train = extract_tensors(training_samples)
        X_dev, Y_dev = extract_tensors(dev_samples)

        # --- 3. CONSTRUIR EL MODELO KERAS ---
        print("Construyendo el modelo...")
        # Entradas
        input_words = Input(shape=(4,), name="input_words")
        input_upos = Input(shape=(4,), name="input_upos")
        
        # Embeddings
        emb_words = Embedding(input_dim=len(self.word_to_id), output_dim=self.word_emb_dim, name="emb_words")(input_words)
        emb_upos = Embedding(input_dim=len(self.upos_to_id), output_dim=20, name="emb_upos")(input_upos) # 20 dim para POS es común
        
        # Aplanar embeddings
        flat_words = Flatten()(emb_words)
        flat_upos = Flatten()(emb_upos)
        
        # Concatenar todas las features
        concatenated = Concatenate()([flat_words, flat_upos])
        
        # Capa oculta
        hidden = Dense(self.hidden_dim, activation='relu', name="hidden_layer")(concatenated)
        
        # Dos salidas separadas (Softmax)
        out_action = Dense(len(self.action_to_id), activation='softmax', name="out_action")(hidden)
        out_deprel = Dense(len(self.deprel_to_id), activation='softmax', name="out_deprel")(hidden)
        
        self.model = Model(inputs=[input_words, input_upos], outputs=[out_action, out_deprel])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                            loss={'out_action': 'sparse_categorical_crossentropy', 
                                'out_deprel': 'sparse_categorical_crossentropy'},
                            metrics={'out_action': 'accuracy', 
                                    'out_deprel': 'accuracy'})

        # --- 4. ENTRENAR EL MODELO ---
        print("Iniciando entrenamiento...")
        self.model.fit(X_train, Y_train, 
                        validation_data=(X_dev, Y_dev),
                        batch_size=self.batch_size, 
                        epochs=self.epochs, 
                        verbose=1)

    def evaluate(self, samples: list['Sample']):
        import numpy as np
        print(f"\nEvaluando el modelo en {len(samples)} muestras...")
        
        # 1. Extraemos los tensores exactamente igual que en el entrenamiento
        X_words = np.zeros((len(samples), 4), dtype='int32')
        X_upos = np.zeros((len(samples), 4), dtype='int32')
        Y_action = np.zeros((len(samples),), dtype='int32')
        Y_deprel = np.zeros((len(samples),), dtype='int32')
        
        for i, s in enumerate(samples):
            feats = s.state_to_feats()
            # Convertimos a IDs (usando <UNK> = 1 si la palabra/POS no se vio en entrenamiento)
            X_words[i] = [self.word_to_id.get(w, 1) for w in feats[:4]]
            X_upos[i] = [self.upos_to_id.get(p, 1) for p in feats[4:]]
            
            # Recogemos las etiquetas reales (gold)
            Y_action[i] = self.action_to_id.get(s.transition.action, 0)
            
            dep = s.transition.dependency if s.transition.dependency else "None"
            Y_deprel[i] = self.deprel_to_id.get(dep, self.deprel_to_id["None"])

        # 2. Hacemos la predicción con nuestro modelo
        predictions = self.model.predict([X_words, X_upos], batch_size=self.batch_size, verbose=0)
        action_probs = predictions[0]
        deprel_probs = predictions[1]

        # 3. Calculamos cuántas veces hemos acertado la acción y la dependencia
        correct_actions = 0
        correct_deprels = 0

        for i in range(len(samples)):
            predicted_action_id = np.argmax(action_probs[i])
            predicted_deprel_id = np.argmax(deprel_probs[i])

            if predicted_action_id == Y_action[i]:
                correct_actions += 1
            if predicted_deprel_id == Y_deprel[i]:
                correct_deprels += 1

        # 4. Imprimimos los resultados
        action_acc = (correct_actions / len(samples)) * 100
        deprel_acc = (correct_deprels / len(samples)) * 100

        print(f"Accuracy de Acción:      {action_acc:.2f}%")
        print(f"Accuracy de Dependencia: {deprel_acc:.2f}%")
    
    def run(self, sents: list['Token']):
        # Importamos las herramientas que vamos a necesitar (¡Ahora con Transition!)
        from .algorithm import ArcEager, State, Sample, Transition
        import numpy as np

        arc_eager = ArcEager()

        for sent in sents:
            for token in sent[1:]:  # Empezamos en 1 para ignorar el ROOT
                token.head = "_"
                token.dep = "_"

        # 1. Initialize: Create the initial state for each sentence.
        batch_states = [arc_eager.create_initial_state(sent) for sent in sents]
        
        # Necesitamos saber a qué oración pertenece cada estado para actualizarlos
        active_indices = list(range(len(sents)))

        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.
        while active_indices:
            
            # 2. Feature Representation: Convert states to their corresponding list of features.
            X_words = np.zeros((len(batch_states), 4), dtype='int32')
            X_upos = np.zeros((len(batch_states), 4), dtype='int32')
            
            for i, state in enumerate(batch_states):
                sample = Sample(state, None)
                feats = sample.state_to_feats()
                
                # Convertimos a IDs
                X_words[i] = [self.word_to_id.get(w, 1) for w in feats[:4]]
                X_upos[i] = [self.upos_to_id.get(p, 1) for p in feats[4:]]

            # 3. Model Prediction: Use the model to predict the next transition and dependency type
            predictions = self.model.predict([X_words, X_upos], batch_size=len(batch_states), verbose=0)
            
            action_probs = predictions[0]  
            deprel_probs = predictions[1]  

            new_batch_states = []
            new_active_indices = []

            for i, state in enumerate(batch_states):
                
                # 4. Transition Sorting: sort the transitions by likelihood
                sorted_action_ids = np.argsort(action_probs[i])[::-1]
                
                # ... and select the most likely dependency type
                best_deprel_id = np.argmax(deprel_probs[i])
                best_deprel_str = self.id_to_deprel.get(best_deprel_id, "None")
                if best_deprel_str == "None":
                    best_deprel_str = None

                # 5. Validation Check: Verify if the selected transition is valid. 
                transition_applied = False
                for action_id in sorted_action_ids:
                    action_str = self.id_to_action[action_id]
                    
                    # Comprobamos la validez según el oráculo (Corregido: usamos Transition directamente)
                    if action_str == arc_eager.LA and arc_eager.LA_is_valid(state):
                        transition = Transition(action_str, best_deprel_str)
                        arc_eager.apply_transition(state, transition)
                        transition_applied = True
                        break
                    elif action_str == arc_eager.RA and arc_eager.RA_is_valid(state):
                        transition = Transition(action_str, best_deprel_str)
                        arc_eager.apply_transition(state, transition)
                        transition_applied = True
                        break
                    elif action_str == arc_eager.REDUCE and arc_eager.REDUCE_is_valid(state):
                        transition = Transition(action_str)
                        arc_eager.apply_transition(state, transition)
                        transition_applied = True
                        break
                    elif action_str == arc_eager.SHIFT:
                        if len(state.B) > 0:
                            transition = Transition(action_str)
                            arc_eager.apply_transition(state, transition)
                            transition_applied = True
                            break
                            
                # Mecanismo de seguridad
                if not transition_applied:
                    if len(state.B) > 0:
                         arc_eager.apply_transition(state, Transition(arc_eager.SHIFT))
                    elif len(state.S) > 0:
                         arc_eager.apply_transition(state, Transition(arc_eager.REDUCE))

                # 6 & 7. State Update & Final State Check
                if not arc_eager.final_state(state):
                    new_batch_states.append(state)
                    new_active_indices.append(active_indices[i])
                else:
                    # Cuando termina, asignamos los arcos al árbol original
                    sent = sents[active_indices[i]]
                    for head_id, rel, dependent_id in state.A:
                         sent[dependent_id].head = head_id
                         sent[dependent_id].dep = rel
                         
            # Actualizamos las listas para la siguiente iteración
            batch_states = new_batch_states
            active_indices = new_active_indices
            
        print("¡Inferencia completada en todas las oraciones!")


if __name__ == "__main__":
    
    model = ParserMLP()