from itertools import product
from typing import Iterable
from pyformlang.rsa import RecursiveAutomaton
from networkx import MultiDiGraph
from pyformlang.finite_automaton import *
from scipy.sparse import dok_matrix, kron
from project.task2 import graph_to_nfa, regex_to_dfa


class FiniteAutomaton:
    def __init__(
        self,
        fa: NondeterministicFiniteAutomaton = None,
        *,
        matrix=None,
        start_states=None,
        final_states=None,
        states_to_int=None,
        states=None,
        from_rsm=False,
    ):
        self.matrix = matrix if fa is None else to_matrix(fa, {v: i for i, v in enumerate(fa.states)})
        self.start_states = start_states if fa is None else fa.start_states
        self.final_states = final_states if fa is None else fa.final_states
        self.states_to_int = states_to_int if fa is None else {v: i for i, v in enumerate(fa.states)}
        self.states = states if fa is None else list(fa.states)
        self.nfa = to_nfa(self) if fa is None and not from_rsm else fa


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
    states = [
        (N.value, state.value)
        for N, box in rsm.boxes.items()
        for state in box.dfa.states
    ]

    n_states = len(states)

    mapping = {state: i for i, state in enumerate(states)}

    start_states = {
        (N.value, start_state.value)
        for N, box in rsm.boxes.items()
        for start_state in box.dfa.start_states
    }

    final_states = {
        (N.value, final_state.value)
        for N, box in rsm.boxes.items()
        for final_state in box.dfa.final_states
    }

    matrix = {}
    for N, box in rsm.boxes.items():
        for from_state, transitions in box.dfa.to_dict().items():
            for symbol, to_state in transitions.items():
                from_idx = mapping[(N.value, from_state.value)]
                to_idx = mapping[(N.value, to_state.value)]
                matrix.setdefault(
                    symbol.value, dok_matrix((n_states, n_states), dtype=bool)
                )[from_idx, to_idx] = True

    return FiniteAutomaton(
        fa=None,
        matrix=matrix,
        start_states=start_states,
        final_states=final_states,
        states_to_int=mapping,
        states=states,
        from_rsm=True,
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
                        State(fa.states_to_int[State(start)]),
                        symbol,
                        State(fa.states_to_int[State(end)]),
                    )

    for state in fa.start_states:
        nfa.add_start_state(State(fa.states_to_int[State(state)]))
    for state in fa.final_states:
        nfa.add_final_state(State(fa.states_to_int[State(state)]))

    return nfa


def intersect_automata(automaton1: FiniteAutomaton, automaton2: FiniteAutomaton) -> FiniteAutomaton:
    """
    Выполняет пересечение двух конечных автоматов.

    Args:
        automaton1 (FiniteAutomaton): Первый конечный автомат.
        automaton2 (FiniteAutomaton): Второй конечный автомат.

    Returns:
        FiniteAutomaton: Результат пересечения.
    """
    matrix = dict()
    start_states = set()
    final_states = set()
    states_to_int = dict()

    for label in automaton1.matrix.keys() & automaton2.matrix.keys():
        matrix[label] = kron(automaton1.matrix[label], automaton2.matrix[label], "csr")

    for state1, int1 in automaton1.states_to_int.items():
        for state2, int2 in automaton2.states_to_int.items():

            combined_int = len(automaton2.states_to_int) * int1 + int2
            states_to_int[combined_int] = combined_int

            if state1 in automaton1.start_states and state2 in automaton2.start_states:
                start_states.add(State(combined_int))

            if state1 in automaton1.final_states and state2 in automaton2.final_states:
                final_states.add(State(combined_int))

    return FiniteAutomaton(
        fa=None,
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

    front = None
    for mat in fa.matrix.values():
        if front is None:
            front = mat
            continue
        front = front + mat
    prev = 0
    while front.count_nonzero() != prev:
        prev = front.count_nonzero()
        front += front @ front

    return front


def paths_ends(graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str) -> list[tuple[object, object]]:
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
    intersect = intersect_automata(graph_fa, regex_fa)
    regex_fa_n = next(iter(regex_fa.matrix.values())).shape[0]

    inter_start_states = {intersect.states_to_int[i] for i in intersect.start_states}
    inter_final_states = {intersect.states_to_int[i] for i in intersect.final_states}

    result = {(graph_fa.states[state // regex_fa_n].value, graph_fa.states[state // regex_fa_n].value)
              for state in inter_start_states & inter_final_states}

    if not intersect.matrix:
        return list(result)

    matrix = sum(intersect.matrix.values())
    matrix = matrix + (matrix @ matrix)

    for _ in range(matrix.shape[0] - 1):
        matrix = matrix + (matrix @ matrix)

    for start, end in product(inter_start_states, inter_final_states):
        if matrix[start, end] != 0:
            result.add((graph_fa.states[start // regex_fa_n].value, graph_fa.states[end // regex_fa_n].value))

    return list(result)