from typing import Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import *
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import dok_matrix, kron
from project.task2 import graph_to_nfa, regex_to_dfa


class FiniteAutomaton:
    lbl = True

    def __init__(
            self,
            fa: NondeterministicFiniteAutomaton = None,
            *,
            matrix=None,
            start_states=None,
            final_states=None,
            states_to_int=None,
            bad_states=False,
            epsilons=None
    ):
        self.matrix = matrix if fa is None else to_matrix(fa, {v: i for i, v in enumerate(fa.states)})
        self.start_states = start_states if fa is None else fa.start_states
        self.final_states = final_states if fa is None else fa.final_states
        self.states_to_int = states_to_int if fa is None else {v: i for i, v in enumerate(fa.states)}
        self.nfa = to_nfa(self) if fa is None and not bad_states else fa
        self.epsilons = epsilons if fa is None else None

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """
        Проверяет, принимает ли автомат заданное слово.

        Args:
            word (Iterable[Symbol]): Слово для проверки.

        Returns:
            bool: True, если автомат принимает слово, иначе False.
        """
        return self.nfa.accepts(word)

    def is_empty(self) -> bool:
        """
        Проверяет, является ли автомат пустым.

        Returns:
            bool: True, если автомат пуст, иначе False.
        """
        return self.nfa.is_empty()

    def labels(self):
        """
        Возвращает множество меток (символов), используемых в автомате.

        Returns:
            set: Множество меток.
        """
        return self.states_to_int.keys() if self.lbl else self.matrix.keys()

    def mapping_indexs(self, u):
        return self.states_to_int[State(u)]

    def labels(self):
        return self.states_to_int.keys() if self.lbl else self.matrix.keys()

    def revert_mapping(self):
        return {i: v for v, i in self.states_to_int.items()}


def to_set(state):
    """
    Преобразует входное состояние в множество, если оно
    уже является множеством, иначе возвращает его как множество.

    Args:
        state: Входное состояние.

    Returns:
        set: Множество состояний.
    """
    return state if isinstance(state, set) else {state}


def to_matrix(fa: NondeterministicFiniteAutomaton, states_to_int=None):
    """
    Преобразует неразрешимый конечный автомат в матрицу переходов.

    Args:
        fa (NondeterministicFiniteAutomaton): Неразрешимый конечный автомат.
        states_to_int (dict, optional): Словарь для преобразования состояний в целые числа. По умолчанию None.

    Returns:
        dict: Словарь, где ключи - это символы, а значения - матрицы переходов.
    """
    result = dict()

    for symbol in fa.symbols:
        result[symbol] = dok_matrix((len(fa.states), len(fa.states)), dtype=bool)
        for start, edges in fa.to_dict().items():
            if symbol in edges:
                for end in to_set(edges[symbol]):
                    result[symbol][states_to_int[start], states_to_int[end]] = True

    return result


def rsm_to_fa(rsm: RecursiveAutomaton) -> FiniteAutomaton:
    states = set()
    start_states = set()
    final_states = set()
    epsilons = set()

    for label, enfa in rsm.boxes.items():
        for state in enfa.dfa.states:
            states.add(State((label, state.value)))
            if state in enfa.dfa.start_states:
                start_states.add(State((label, state.value)))
            if state in enfa.dfa.final_states:
                final_states.add(State((label, state.value)))

    states_to_int = {s: i for i, s in enumerate(states)}

    matrix = dict()
    for label, enfa in rsm.boxes.items():
        for frm, transition in enfa.dfa.to_dict().items():
            for symbol, to in transition.items():
                if symbol not in matrix:
                    matrix[symbol.value] = dok_matrix((len(states), len(states)), dtype=bool)
                for target in to_set(to):
                    matrix[symbol.value][
                        states_to_int[State((label, frm.value))],
                        states_to_int[State((label, target.value))],
                    ] = True
                if isinstance(to, Epsilon):
                    epsilons.add(label)

    return FiniteAutomaton(
        fa=None,
        matrix=matrix,
        start_states=start_states,
        final_states=final_states,
        states_to_int=states_to_int,
        bad_states=True,
        epsilons=epsilons,
    )


def to_nfa(fa: FiniteAutomaton) -> NondeterministicFiniteAutomaton:
    """
    Преобразует конечный автомат в неразрешимый конечный автомат.

    Args:
        fa (FiniteAutomaton): Конечный автомат.

    Returns:
        NondeterministicFiniteAutomaton: Неразрешимый конечный автомат.
    """
    nfa = NondeterministicFiniteAutomaton()

    for symbol in fa.matrix.keys():
        for start in range(fa.matrix[symbol].shape[0]):
            for end in range(fa.matrix[symbol].shape[0]):
                if fa.matrix[symbol][start, end]:
                    nfa.add_transition(
                        State(start),
                        symbol,
                        State(end)
                    )

    for state in fa.start_states:
        nfa.add_start_state(State(state))
    for state in fa.final_states:
        nfa.add_final_state(State(state))

    return nfa


def intersect_automata(
        automaton1: FiniteAutomaton, automaton2: FiniteAutomaton, lbl=True
) -> FiniteAutomaton:
    """
    Выполняет пересечение двух конечных автоматов.

    Args:
        automaton1 (FiniteAutomaton): Первый конечный автомат.
        automaton2 (FiniteAutomaton): Второй конечный автомат.
        lbl (bool, optional): Флаг, указывающий, следует ли использовать метки. По умолчанию True.

    Returns:
        FiniteAutomaton: Результат пересечения.
    """
    automaton1.lbl = automaton2.lbl = not lbl
    labels = automaton1.labels() & automaton2.labels()
    matrix = dict()
    start_states = set()
    final_states = set()
    states_to_int = dict()

    for label in labels:
        matrix[label] = kron(automaton1.matrix[label], automaton2.matrix[label], "csr")

    for state1, int1 in automaton1.states_to_int.items():
        for state2, int2 in automaton2.states_to_int.items():
            combined_state = (state1, state2)
            combined_int = len(automaton2.states_to_int) * int1 + int2
            states_to_int[combined_state] = combined_int

            if state1 in automaton1.start_states and state2 in automaton2.start_states:
                start_states.add(State(combined_int))

            if state1 in automaton1.final_states and state2 in automaton2.final_states:
                final_states.add(State(combined_int))

    return FiniteAutomaton(
        matrix=matrix,
        start_states=start_states,
        final_states=final_states,
        states_to_int=states_to_int,
    )


def transitive_closure(fa: FiniteAutomaton):
    """
    Вычисляет транзитивное замкнутость матрицы переходов конечного автомата.

    Args:
        fa (FiniteAutomaton): Конечный автомат.

    Returns:
        dok_matrix: Матрица транзитивного замкнутости.
    """
    if len(fa.matrix.values()) == 0:
        return dok_matrix((0, 0), dtype=bool)

    front = sum(fa.matrix.values(), dok_matrix((0, 0), dtype=bool))
    prev = 0
    while front.count_nonzero() != prev:
        prev = front.count_nonzero()
        front += front @ front

    return front


def paths_ends(
        graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[object, object]]:
    """
    Находит пути в графе, которые заканчиваются в конечных узлах и
    соответствуют заданному регулярному выражению.

    Args:
        graph (MultiDiGraph): Граф.
        start_nodes (set[int]): Начальные узлы.
        final_nodes (set[int]): Конечные узлы.
        regex (str): Регулярное выражение.

    Returns:
        list[tuple[object, object]]: Список кортежей, представляющих пути.
    """
    graph_fa = FiniteAutomaton(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = FiniteAutomaton(regex_to_dfa(regex))
    intersect = intersect_automata(graph_fa, regex_fa, lbl=False)

    closure = transitive_closure(intersect)
    reg_size = len(regex_fa.states_to_int)
    result = list()
    for start, end in zip(*closure.nonzero()):
        if start in intersect.start_states and end in intersect.final_states:
            result.append(
                (
                    graph_fa.states_to_int[start // reg_size],
                    graph_fa.states_to_int[end // reg_size],
                )
            )
    return result
