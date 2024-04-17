from pyformlang.finite_automaton import *
from scipy.sparse import *
from networkx import *
from typing import *

from project.finite_automatons import regex_to_dfa, graph_to_nfa

from itertools import product


class FiniteAutomaton:
    def __init__(self, nondeterministic_automaton=None):
        """
        Конструктор класса FiniteAutomaton.

        Параметры:
        - nka: Объект DeterministicFiniteAutomaton или NondeterministicFiniteAutomaton,
               который будет преобразован в конечный автомат.

        Пример использования:
        >>> automaton = FiniteAutomaton(nka)
        """
        if nondeterministic_automaton is None:
            return

        # Создаем словарь для отображения индексов состояний на сами состояния
        state_index_map = {
            state: index
            for index, state in enumerate(nondeterministic_automaton.states)
        }

        # Сохраняем список состояний
        self.state_index_map = list(nondeterministic_automaton.states)

        # Сохраняем начальные и конечные состояния
        self.start_states = {
            state_index_map[st] for st in nondeterministic_automaton.start_states
        }
        self.final_states = {
            state_index_map[fi] for fi in nondeterministic_automaton.final_states
        }

        # Создаем словарь для хранения функций переходов
        self.transition_functions = {}

        # Получаем словарь переходов из nka
        states_dict = nondeterministic_automaton.to_dict()
        num_states = len(nondeterministic_automaton.states)

        # Заполняем словарь функций переходов
        for symbols in nondeterministic_automaton.symbols:
            self.transition_functions[symbols] = dok_matrix(
                (num_states, num_states), dtype=bool
            )
            for current_state, next_state_set in states_dict.items():
                if symbols in next_state_set:
                    for next_state in (
                        next_state_set[symbols]
                        if isinstance(next_state_set[symbols], set)
                        else {next_state_set[symbols]}
                    ):
                        self.transition_functions[symbols][
                            state_index_map[current_state], state_index_map[next_state]
                        ] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """
        Проверяет, принимает ли автомат заданное слово.

        Параметры:
        - word: Итерируемый объект, содержащий символы, которые должны быть проверены.

        Возвращает:
        - bool: True, если автомат принимает слово, иначе False.

        Пример использования:
        >>> automaton.accepts(['a', 'b', 'c'])
        True
        """

        temp_nondeterministic_automaton = NondeterministicFiniteAutomaton()

        # Добавляем переходы в nka
        for transition_symbol, transition_matrix in self.transition_functions.items():
            temp_nondeterministic_automaton.add_transitions(
                [
                    (start_state, transition_symbol, end_state)
                    for (start_state, end_state) in product(
                        range(transition_matrix.shape[0]), repeat=2
                    )
                    if self.transition_functions[transition_symbol][
                        start_state, end_state
                    ]
                ]
            )

        # Добавляем начальные и конечные состояния в nka
        for start_state in self.start_states:
            temp_nondeterministic_automaton.add_start_state(start_state)
        for final_state in self.final_states:
            temp_nondeterministic_automaton.add_final_state(final_state)

        # Проверяем, принимает ли nka заданное слово
        return temp_nondeterministic_automaton.accepts(word)

    # Метод для проверки, является ли автомат пустым
    def is_empty(self) -> bool:
        """
        Проверяет, является ли язык, задающийся автоматом, пустым.

        Возвращает:
        - bool: True, если язык пуст, иначе False.

        Пример использования:
        >>> automaton.is_empty()
        False
        """
        # Если в автомате нет функций переходов, он считается пустым
        if len(self.transition_functions) == 0:
            return True

        # Создаем матрицу, которая представляет все переходы в автомате
        transition_matrix = sum(self.transition_functions.values())

        # Выполняем операцию Крона для получения матрицы переходов
        for _ in range(transition_matrix.shape[0]):
            transition_matrix += transition_matrix @ transition_matrix

        return not any(
            transition_matrix[start_state, final_state]
            for start_state, final_state in product(
                self.start_states, self.final_states
            )
        )


def intersect_automata(
    automaton1: FiniteAutomaton, automaton2: FiniteAutomaton
) -> FiniteAutomaton:
    """
    Выполняет пересечение двух конечных автоматов.

    Параметры:
    - automaton1: Первый конечный автомат.
    - automaton2: Второй конечный автомат.

    Возвращает:
    - FiniteAutomaton: Новый конечный автомат, представляющий пересечение двух входных автоматов.

    Пример использования:
    >>> new_automaton = intersect_automata(automaton1, automaton2)
    """
    # Находим общие ключи для функций переходов обоих автоматов
    common_keys = (
        automaton1.transition_functions.keys() & automaton2.transition_functions.keys()
    )
    new_automaton = FiniteAutomaton()
    new_automaton.transition_functions = {}

    # Для каждого общего ключа выполняем операцию Крона
    for key in common_keys:
        new_automaton.transition_functions[key] = kron(
            automaton1.transition_functions[key],
            automaton2.transition_functions[key],
            "csr",
        )

    # Определяем начальные и конечные состояния нового автомата
    new_automaton.start_states = set()
    new_automaton.final_states = set()

    num_states2 = (
        automaton2.transition_functions.values().__iter__().__next__().shape[0]
    )

    for state1, state2 in product(automaton1.start_states, automaton2.start_states):
        new_automaton.start_states.add(state1 * (num_states2) + state2)

    for state1, state2 in product(automaton1.final_states, automaton2.final_states):
        new_automaton.final_states.add(state1 * (num_states2) + state2)

    return new_automaton


def paths_ends(
    graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[int, int]]:
    """
    Находит пары вершин в графе, которые связаны путем, формирующим слово из языка, задаваемого регулярным выражением.

    Параметры:
    - graph: Объект MultiDiGraph, представляющий граф.
    - start_nodes: Множество начальных вершин графа.
    - final_nodes: Множество конечных вершин графа.
    - regex: Регулярное выражение, задающее язык.

    Возвращает:
    - list[tuple[int, int]]: Список пар вершин, которые связаны путем, формирующим слово из языка, задаваемого регулярным выражением.

    Пример использования:
    >>> paths = paths_ends(graph, {1, 2}, {3, 4}, 'a*b')
    [(1, 3), (2, 4)]
    """
    # Создаем конечный автомат из графа, используя функцию graph_to_nfa,
    # которая преобразует граф в недетерминированный конечный автомат (NFA).
    finite_automaton_from_graph = FiniteAutomaton(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )

    # Создаем конечный автомат из регулярного выражения, используя функцию regex_to_dfa,
    # которая преобразует регулярное выражение в детерминированный конечный автомат (DFA).
    finite_automaton_from_regex = FiniteAutomaton(regex_to_dfa(regex))

    # Выполняем пересечение двух автоматов, полученных на предыдущих шагах.
    intersected_automaton = intersect_automata(
        finite_automaton_from_graph, finite_automaton_from_regex
    )

    # Если в полученном автомате нет функций переходов, возвращаем пустой список.
    if not intersected_automaton.transition_functions:
        return []

    # Создаем матрицу, которая представляет все переходы в полученном автомате.
    transition_matrix = sum(intersected_automaton.transition_functions.values())

    # Выполняем операцию Крона для получения матрицы переходов.
    for _ in range(transition_matrix.shape[0]):
        transition_matrix += transition_matrix @ transition_matrix

    # Определяем количество состояний в автомате, соответствующем регулярному выражению.
    n_states2 = (
        finite_automaton_from_regex.transition_functions.values()
        .__iter__()
        .__next__()
        .shape[0]
    )

    # Функция для преобразования индексов состояний в узлы графа.
    convert_to_node = lambda i: finite_automaton_from_graph.state_index_map[
        i // n_states2
    ].value

    # Используем функцию product из модуля itertools для генерации декартового произведения
    # начальных и конечных состояний полученного автомата.
    # Для каждой пары состояний проверяем, существует ли путь между ними,
    # используя матрицу переходов m. Если путь существует, добавляем его в результат.
    return [
        (convert_to_node(st), convert_to_node(fi))
        for st, fi in product(
            intersected_automaton.start_states, intersected_automaton.final_states
        )
        if transition_matrix[st, fi] != 0
    ]
