from .conllu_token import Token
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 200, 
                epochs: int = 20, batch_size: int = 64):
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.upos_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.lemma_to_id = {"<PAD>": 0, "<UNK>": 1} # Nuevo vocabulario
        
        self.action_to_id = {}
        self.deprel_to_id = {}
        self.id_to_action = {}
        self.id_to_deprel = {}
        self.model = None
    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        import numpy as np

        # --- 1. CONSTRUIR DICCIONARIOS (Incluyendo Lemmas y .lower()) ---
        print("Construyendo vocabularios...")
        for sample in training_samples:
            feats = sample.state_to_feats()
            # 0-3: Words, 4-7: UPOS, 8-11: Lemmas
            for word in feats[0:4]:
                w = word.lower()
                if w not in self.word_to_id: self.word_to_id[w] = len(self.word_to_id)
            for pos in feats[4:8]:
                if pos not in self.upos_to_id: self.upos_to_id[pos] = len(self.upos_to_id)
            for lemma in feats[8:12]:
                l = lemma.lower()
                if l not in self.lemma_to_id: self.lemma_to_id[l] = len(self.lemma_to_id)
            
            action = sample.transition.action
            if action not in self.action_to_id:
                idx = len(self.action_to_id)
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action
            
            dep = sample.transition.dependency or "None"
            if dep not in self.deprel_to_id:
                idx = len(self.deprel_to_id)
                self.deprel_to_id[dep] = idx
                self.id_to_deprel[idx] = dep

        # --- 2. PREPARAR DATOS (3 Inputs ahora) ---
        def extract_tensors(samples):
            size = len(samples)
            Xw, Xp, Xl = np.zeros((size, 4)), np.zeros((size, 4)), np.zeros((size, 4))
            Ya, Yd = np.zeros((size,)), np.zeros((size,))
            for i, s in enumerate(samples):
                f = s.state_to_feats()
                Xw[i] = [self.word_to_id.get(w.lower(), 1) for w in f[0:4]]
                Xp[i] = [self.upos_to_id.get(p, 1) for p in f[4:8]]
                Xl[i] = [self.lemma_to_id.get(l.lower(), 1) for l in f[8:12]]
                Ya[i] = self.action_to_id[s.transition.action]
                dep = s.transition.dependency or "None"
                Yd[i] = self.deprel_to_id.get(dep, self.deprel_to_id["None"])
            return [Xw, Xp, Xl], [Ya, Yd]

        X_train, Y_train = extract_tensors(training_samples)
        X_dev, Y_dev = extract_tensors(dev_samples)

        # --- 3. ARQUITECTURA (hidden_dim=200, Dropout, UPOS=32) ---
        in_w = Input(shape=(4,), name="input_words")
        in_p = Input(shape=(4,), name="input_upos")
        in_l = Input(shape=(4,), name="input_lemmas")
        
        emb_w = Embedding(len(self.word_to_id), self.word_emb_dim)(in_w)
        emb_p = Embedding(len(self.upos_to_id), 32)(in_p) # Subido a 32
        emb_l = Embedding(len(self.lemma_to_id), 100)(in_l)
        
        flat = Concatenate()([Flatten()(emb_w), Flatten()(emb_p), Flatten()(emb_l)])
        
        # Dos capas densas con Dropout para frenar el Overfitting
        x = Dense(self.hidden_dim, activation='relu')(flat)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x) # Capa extra
        x = Dropout(0.3)(x)
        
        out_a = Dense(len(self.action_to_id), activation='softmax', name="out_action")(x)
        out_d = Dense(len(self.deprel_to_id), activation='softmax', name="out_deprel")(x)
        
        self.model = Model(inputs=[in_w, in_p, in_l], outputs=[out_a, out_d])
        self.model.compile(optimizer=Adam(0.001), 
                           loss='sparse_categorical_crossentropy',
                           metrics={'out_action': 'accuracy', 'out_deprel': 'accuracy'})

        # --- 4. ENTRENAMIENTO CON EARLY STOPPING ---
        # Usamos val_loss (suma de ambos errores) y paciencia 5
        # Esto asegura que no pare hasta que AMBAS tareas dejen de mejorar
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev),
                       batch_size=self.batch_size, epochs=self.epochs, 
                       callbacks=[es], verbose=1)

    def evaluate(self, samples: list['Sample']):
        print(f"\nEvaluando el modelo en {len(samples)} muestras...")
        
        # 1. Preparamos TRES tensores de entrada y los dos de salida
        size = len(samples)
        X_words = np.zeros((size, 4), dtype='int32')
        X_upos = np.zeros((size, 4), dtype='int32')
        X_lemmas = np.zeros((size, 4), dtype='int32') # <-- NUEVO
        
        Y_action = np.zeros((size,), dtype='int32')
        Y_deprel = np.zeros((size,), dtype='int32')
        
        for i, s in enumerate(samples):
            feats = s.state_to_feats()
            # Mapeo a IDs con .lower() para Words y Lemmas (igual que en train)
            X_words[i] = [self.word_to_id.get(w.lower(), 1) for w in feats[0:4]]
            X_upos[i] = [self.upos_to_id.get(p, 1) for p in feats[4:8]]
            X_lemmas[i] = [self.lemma_to_id.get(l.lower(), 1) for l in feats[8:12]] # <-- NUEVO
            
            # Etiquetas reales
            Y_action[i] = self.action_to_id.get(s.transition.action, 0)
            dep = s.transition.dependency or "None"
            Y_deprel[i] = self.deprel_to_id.get(dep, self.deprel_to_id["None"])

        # 2. Predicción pasando los TRES tensores
        predictions = self.model.predict([X_words, X_upos, X_lemmas], 
                                         batch_size=self.batch_size, 
                                         verbose=0)
        action_probs = predictions[0]
        deprel_probs = predictions[1]

        # 3. Cálculo de aciertos
        correct_actions = 0
        correct_deprels = 0

        for i in range(size):
            p_action = np.argmax(action_probs[i])
            p_deprel = np.argmax(deprel_probs[i])

            if p_action == Y_action[i]:
                correct_actions += 1
            if p_deprel == Y_deprel[i]:
                correct_deprels += 1

        action_acc = (correct_actions / size) * 100
        deprel_acc = (correct_deprels / size) * 100

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
            X_lemmas = np.zeros((len(batch_states), 4), dtype='int32')
            
            for i, state in enumerate(batch_states):
                sample = Sample(state, None)
                feats = sample.state_to_feats()
                X_words[i] = [self.word_to_id.get(w.lower(), 1) for w in feats[0:4]]
                X_upos[i] = [self.upos_to_id.get(p, 1) for p in feats[4:8]]
                X_lemmas[i] = [self.lemma_to_id.get(l.lower(), 1) for l in feats[8:12]]

            # 3. Model Prediction: Use the model to predict the next transition and dependency type
            predictions = self.model.predict([X_words, X_upos, X_lemmas], batch_size=len(batch_states), verbose=0)
            
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