from .state import State
from .conllu_token import Token


class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                                 parser action to be taken in the given state.

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Transition):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition
    

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Extracts features from a given parsing state for use in a transition-based dependency parser.

        This function generates a feature representation from the current state of the parser, 
        which includes features from both the stack and the buffer. The number of features from 
        the stack and the buffer can be specified.

        Parameters:
            nbuffer_feats (int): The number of features to extract from the buffer.
            nstack_feats (int): The number of features to extract from the stack.

        Returns:
            list[str]: A list of extracted features. The features include the words and their 
                    corresponding UPOS (Universal Part-of-Speech) tags from the specified number 
                    of elements in the stack and buffer. The format of the feature list is as follows:
                    [Word_stack_n,...,Word_stack_0, Word_buffer_0,...,Word_buffer_m, 
                        UPOS_stack_n,...,UPOS_stack_0, UPOS_buffer_0,...,UPOS_buffer_m]
                    where 'n' is nstack_feats and 'm' is nbuffer_feats.

        Examples:
            Example 1:
                State: Stack (size=1): (0, ROOT, ROOT_UPOS)
                    Buffer (size=13): (1, Distribution, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=0): []

                Output: ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']

            Example 2:
                State: Stack (size=2): (0, ROOT, ROOT_UPOS) | (1, Distribution, NOUN)
                    Buffer (size=10): (4, license, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=2): [(4, 'det', 3), (4, 'case', 2)]

                Output: ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
        """
        state = self._state
        feats_words = []
        feats_tags = []

        # 1. Stack (S0, S1) y sus hijos (Rasgos de 2º orden)
        stack_tokens = state.S[::-1]
        for i in range(nstack_feats):
            if i < len(stack_tokens):
                t = stack_tokens[i]
                feats_words.append(t.form.lower()) # Normalización
                feats_tags.append(t.upos)
                
                # Buscamos hijos para dar contexto estructural[cite: 1]
                hijos = [arc for arc in state.A if arc[0] == t.id]
                if hijos:
                    lc_tag = min(hijos, key=lambda x: x[2])[1] # Hijo más a la izq
                    rc_tag = max(hijos, key=lambda x: x[2])[1] # Hijo más a la der
                    feats_tags.extend([lc_tag, rc_tag])
                else:
                    feats_tags.extend(["<NONE>", "<NONE>"])
            else:
                feats_words.append("<PAD>")
                feats_tags.extend(["<PAD>", "<PAD>", "<PAD>"])

        # 2. Buffer (B0, B1)[cite: 1]
        buffer_tokens = state.B
        for i in range(nbuffer_feats):
            if i < len(buffer_tokens):
                t = buffer_tokens[i]
                feats_words.append(t.form.lower())
                feats_tags.append(t.upos)
            else:
                feats_words.append("<PAD>")
                feats_tags.append("<PAD>")

        return feats_words + feats_tags

    def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"



class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        """
        Precondiciones LEFT-ARC: 
        1. La cima del stack no es el ROOT (id 0).
        2. La cima del stack no tiene ya un padre asignado[cite: 1, 2].
        """
        if not state.S or state.S[-1].id == 0:
            return False
        
        # Comprobar si la cima del stack ya es dependiente (hijo) en algún arco
        for (head, dep, dependent) in state.A:
            if dependent == state.S[-1].id:
                return False
        return True

    def LA_is_correct(self, state: State, gold_arcs: set) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is the correct action for the current parsing state.

        This method checks if applying a LEFT-ARC transition will correctly reflect the dependency
        structure of the sentence being parsed, based on the current state of the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is the correct action in the current state, False otherwise.
        """
        
        """Es correcta si en el árbol real el frente del buffer es padre de la cima del stack[cite: 1]."""
        if not state.S or not state.B:
            return False
        return (state.B[0].id, state.S[-1].id) in [(arc[0], arc[2]) for arc in gold_arcs]
    
    def RA_is_correct(self, state: State, gold_arcs: set) -> bool:
        """
        Determines if a RIGHT-ARC (RA) transition is the correct action for the current parsing state.

        This method assesses whether applying a RIGHT-ARC transition aligns with the correct 
        dependency structure of the sentence, based on the parser's current state.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a RIGHT-ARC transition is the correct action in the current state, False otherwise.
        """
        
        """Es correcta si en el árbol real la cima del stack es padre del frente del buffer[cite: 1]."""
        if not state.S or not state.B:
            return False
        return (state.S[-1].id, state.B[0].id) in [(arc[0], arc[2]) for arc in gold_arcs]

    def RA_is_valid(self, state: State) -> bool:
        """
        Checks the preconditions in order to apply a right-arc (RA) transition.

        A RIGHT-ARC transition is valid under certain conditions related to the state of the stack
        and buffer in the parser. This method evaluates these conditions to determine if a RIGHT-ARC
        can be applied.

        Parameters:
            state (State): The current parsing state of the parser.

        Returns:
            bool: True if a RIGHT-ARC transition can be validly applied in the current state, False otherwise.
        """
        
        """Precondición RIGHT-ARC: El frente del buffer no tiene padre asignado."""
        if not state.B:
            return False
        
        for (head, dep, dependent) in state.A:
            if dependent == state.B[0].id:
                return False
        return True

    def REDUCE_is_correct(self, state: State, gold_arcs: set) -> bool:
        """
        Determines if applying a REDUCE transition is the correct action for the current parsing state.

        A REDUCE transition is correct if there is no word in the buffer (state.B) whose head 
        is the word on the top of the stack (state.S[-1]). This method checks this condition 
        against the current state of the parser.

        REDUCE can be only correct iff for every word in the buffer, 
        no word has as head the top word from the stack 

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is the correct action in the current state, False otherwise.
        """
        
        """
        Es correcta si la cima del stack ya tiene su padre y no tiene hijos pendientes en el buffer[cite: 1].
        """
        if not state.S or not state.B:
            return False
        
        s_id = state.S[-1].id
        # Comprobar si algún nodo en el buffer tiene como padre a la cima del stack
        for (gold_head, gold_dep, gold_dependent) in gold_arcs:
            if gold_head == s_id:
                # Si el hijo (gold_dependent) está todavía en el buffer, no podemos reducir
                if any(b_token.id == gold_dependent for b_token in state.B):
                    return False
        return True

    def REDUCE_is_valid(self, state: State) -> bool:
        """
        Determines if a REDUCE transition is valid for the current parsing state.

        This method checks if the preconditions for applying a REDUCE transition are met in 
        the current state of the parser. This typically involves assessing the state of the 
        stack and buffer.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is valid in the current state, False otherwise.
        """
        
        """Precondición REDUCE: La cima del stack ya tiene un padre asignado[cite: 1, 2]."""
        if not state.S:
            return False
        
        for (head, dep, dependent) in state.A:
            if dependent == state.S[-1].id:
                return True
        return False

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step, given an input dependency tree.

        This function iterates through a given sentence, represented as a dependency tree, to generate a sequence 
        of gold-standard transitions. These transitions are what an ideal parser should predict at each step to 
        correctly parse the sentence. The function checks the validity and correctness of possible transitions 
        at each step and selects the appropriate one based on the arc-eager parsing algorithm. It is primarily 
        used for later training a dependency parser.

        Parameters:
            sent (list['Token']): A list of 'Token' instances representing a dependency tree. Each 'Token' 
                        should contain information about a word/token in a sentence.

        Returns:
            samples (list['Sample']): A list of Sample instances. Each Sample stores an state instance and a transition instance
            with the information of the outputs to predict (the transition and optionally the dependency label)
        """

        state = self.create_initial_state(sent)
        gold = self.gold_arcs(sent)
        samples = []

        while not self.final_state(state):
            # Obtener etiquetas de dependencia para LA y RA del árbol gold
            s_id = state.S[-1].id if state.S else -1
            b_id = state.B[0].id if state.B else -1
            
            # Buscar etiqueta en gold_arcs
            la_dep = next((arc[1] for arc in gold if arc[0] == b_id and arc[2] == s_id), None)
            ra_dep = next((arc[1] for arc in gold if arc[0] == s_id and arc[2] == b_id), None)

            if self.LA_is_valid(state) and self.LA_is_correct(state, gold):
                trans = Transition(self.LA, la_dep)
                samples.append(Sample(state, trans))
                self.apply_transition(state, trans)

            elif self.RA_is_valid(state) and self.RA_is_correct(state, gold):
                trans = Transition(self.RA, ra_dep)
                samples.append(Sample(state, trans))
                self.apply_transition(state, trans)

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state, gold):
                trans = Transition(self.REDUCE)
                samples.append(Sample(state, trans))
                self.apply_transition(state, trans)
            
            else:
                trans = Transition(self.SHIFT)
                samples.append(Sample(state, trans))
                self.apply_transition(state, trans)

        assert gold == state.A, "Error: Los arcos generados no coinciden con los Gold"[cite: 1]
        return samples         
    

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies a given transition to the current parsing state.

        This method updates the state based on the type of transition - LEFT-ARC, RIGHT-ARC, 
        or REDUCE - and the validity of applying such a transition in the current context.

        Parameters:
            state (State): The current parsing state, which includes a stack (S), 
                        a buffer (B), and a set of arcs (A).
            transition (Transition): The transition to be applied, consisting of an action
                                    (LEFT-ARC, RIGHT-ARC, REDUCE) and an optional dependency label (only for LEFT-ARC and RIGHT-arc).

        Returns:
            None; the state is modified in place.
        """

        """Aplica la transición modificando el estado in-place[cite: 1, 2, 4]."""
        t = transition.action
        dep = transition.dependency
        s = state.S[-1] if state.S else None
        b = state.B[0] if state.B else None

        if t == self.LA:
            state.A.add((b.id, dep, s.id)) # Arco: Buffer -> Stack[cite: 1, 2]
            state.S.pop() # Eliminar cima del stack[cite: 1, 2]

        elif t == self.RA:
            state.A.add((s.id, dep, b.id)) # Arco: Stack -> Buffer[cite: 1, 2]
            state.S.append(b) # Mover buffer a stack[cite: 1, 2]
            state.B.pop(0)

        elif t == self.REDUCE:
            state.S.pop() # Eliminar cima del stack[cite: 1, 2]

        else: # SHIFT
            state.S.append(b)
            state.B.pop(0)
    


    def gold_arcs(self, sent: list['Token']) -> set:
        """
        Extracts and returns the gold-standard dependency arcs from a given sentence.

        This function processes a sentence represented by a list of Token objects to extract the dependency relations 
        (arcs) present in the sentence. Each Token object should contain information about its head (the id of the 
        parent token in the dependency tree), the type of dependency, and its own id. The function constructs a set 
        of tuples, each representing a dependency arc in the sentence.

        Parameters:
            sent (list[Token]): A list of Token objects representing the sentence. Each Token object contains 
                                information about a word or punctuation in a sentence, including its dependency 
                                relations and other annotations.

        Returns:
            gold_arcs (set[tuple]): A set of tuples, where each tuple is a triplet (head_id, dependency_type, dependent_id). 
                                    This represents all the gold-standard dependency arcs in the sentence. The head_id and 
                                    dependent_id are integers representing the respective tokens in the sentence, and 
                                    dependency_type is a string indicating the type of dependency relation.
        """
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))

        return gold_arcs


   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)