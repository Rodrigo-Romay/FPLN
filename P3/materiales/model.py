from .conllu_token import Token
import tensorflow as tf
import numpy as np
from .algorithm import ArcEager, Transition

class ParserMLP:
    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 256, 
                 epochs: int = 30, batch_size: int = 128):
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
        # Índices 0 y 1 reservados para PAD y Desconocidos (UNK)[cite: 8].
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.tag2id = {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2}
        self.action2id = {}
        self.id2action = {}
        self.dep2id = {"None": 0}
        self.id2dep = {0: "None"}

    def _build_vocab(self, samples):
        """Filtra palabras de frecuencia 1 para mejorar el aprendizaje[cite: 8]."""
        word_counts = {}
        for s in samples:
            # Obtenemos las 4 palabras de stack y buffer del estado[cite: 2].
            for w in s.state_to_feats()[:4]:
                if w != "<PAD>":
                    word_counts[w] = word_counts.get(w, 0) + 1
            
            # Mapeamos etiquetas UPOS (incluyendo la del hijo de segundo orden)[cite: 1].
            for t in s.state_to_feats()[4:]:
                if t not in self.tag2id:
                    self.tag2id[t] = len(self.tag2id)

            # Mapeamos acciones y dependencias[cite: 2].
            act, dep = s.transition.action, str(s.transition.dependency)
            if act not in self.action2id:
                idx = len(self.action2id)
                self.action2id[act], self.id2action[idx] = idx, act
            if dep not in self.dep2id:
                idx = len(self.dep2id)
                self.dep2id[dep], self.id2dep[idx] = idx, dep

        # Solo añadimos palabras que aparecen al menos 2 veces[cite: 8].
        for word, count in word_counts.items():
            if count > 1 and word not in self.word2id:
                self.word2id[word] = len(self.word2id)

    # --- Dentro de ParserMLP en materiales/model.py ---

    def train(self, training_samples, dev_samples):
        self._build_vocab(training_samples)
        
        # SOLUCIÓN AL ERROR: Calcular pesos por muestra (sample_weight)
        from sklearn.utils import class_weight
        y_actions_all = [self.action2id[s.transition.action] for s in training_samples]
        cw = class_weight.compute_class_weight('balanced', 
                                              classes=np.unique(y_actions_all), 
                                              y=y_actions_all)
        # Creamos un array donde cada muestra tiene el peso de su clase[cite: 8]
        weights_array = np.array([cw[act_id] for act_id in y_actions_all])

        def prepare_data(samples):
            x_w, x_t, y_a, y_d = [], [], [], []
            for s in samples:
                f = s.state_to_feats()
                x_w.append([self.word2id.get(w, 1) for w in f[:4]])
                x_t.append([self.tag2id.get(t, 1) for t in f[4:]])
                y_a.append(self.action2id[s.transition.action])
                y_d.append(self.dep2id.get(str(s.transition.dependency), 0))
            return [np.array(x_w), np.array(x_t)], [np.array(y_a), np.array(y_d)]

        x_train, y_train = prepare_data(training_samples)
        x_dev, y_dev = prepare_data(dev_samples)

        # Arquitectura robusta[cite: 2, 8]
        # Dentro de model.py, en el método train:
        # En lugar de shape=(5,), usa la forma real de los datos preparados:
        in_word = tf.keras.Input(shape=(x_train[0].shape[1],), name="input_words")
        in_tag = tf.keras.Input(shape=(x_train[1].shape[1],), name="input_tags")
        
        emb_w = tf.keras.layers.Embedding(len(self.word2id), self.word_emb_dim)(in_word)
        emb_t = tf.keras.layers.Embedding(len(self.tag2id), self.word_emb_dim)(in_tag)
        
        flat_w = tf.keras.layers.Flatten()(emb_w)
        flat_t = tf.keras.layers.Flatten()(emb_t)
        merged = tf.keras.layers.Concatenate()([flat_w, flat_t])
        
        dense = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(merged)
        drop = tf.keras.layers.Dropout(0.4)(dense) # Evita que se estaque en el 30%[cite: 8]
        dense2 = tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu')(drop)

        out_act = tf.keras.layers.Dense(len(self.action2id), activation='softmax', name='action')(dense2)
        out_dep = tf.keras.layers.Dense(len(self.dep2id), activation='softmax', name='deprel')(dense2)
        
        self.model = tf.keras.Model(inputs=[in_word, in_tag], outputs=[out_act, out_dep])
        
        # Learning rate bajo para estabilidad en Multi-task[cite: 8, 11]
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy', 'accuracy'])
        
        print("Entrenando con pesos balanceados por muestra...")
        # Usamos sample_weight en lugar de class_weight[cite: 8]
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                       validation_data=(x_dev, y_dev), sample_weight=weights_array)

    def run(self, sents):
        ae = ArcEager()
        active = [{"state": ae.create_initial_state(s), "sent": s} for s in sents]
        
        while active:
            x_w, x_t = [], []
            for inst in active:
                from .algorithm import Sample
                f = Sample(inst["state"], None).state_to_feats()
                x_w.append([self.word2id.get(w, 1) for w in f[:4]])
                x_t.append([self.tag2id.get(t, 1) for t in f[4:]])
            
            p_a, p_d = self.model.predict([np.array(x_w), np.array(x_t)], verbose=0)
            
            next_active = []
            for i, inst in enumerate(active):
                state = inst["state"]
                # Convertimos el índice de NumPy a entero de Python para evitar KeyErrors[cite: 8].
                best_dep = self.id2dep[int(np.argmax(p_d[i]))]
                sorted_acts = np.argsort(p_a[i])[::-1]
                
                applied = False
                for a_id in sorted_acts:
                    name = self.id2action[a_id]
                    # Validación de precondiciones antes de aplicar[cite: 1, 2].
                    if (name == ae.SHIFT and state.B) or \
                       (name == ae.LA and ae.LA_is_valid(state)) or \
                       (name == ae.RA and ae.RA_is_valid(state)) or \
                       (name == ae.REDUCE and ae.REDUCE_is_valid(state)):
                        ae.apply_transition(state, Transition(name, best_dep if name in [ae.LA, ae.RA] else None))
                        applied = True
                        break
                
                if applied and not ae.final_state(state):
                    next_active.append(inst)
                else:
                    # Al finalizar, asignamos los arcos al árbol original[cite: 1, 4].
                    for h, d, c in state.A:
                        inst["sent"][c].head, inst["sent"][c].dep = h, d
            active = next_active
        return sents

if __name__ == "__main__":
    
    model = ParserMLP()