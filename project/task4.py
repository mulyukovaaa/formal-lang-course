from project.task3 import FiniteAutomaton
from scipy.sparse import dok_matrix, block_diag


def reachability_with_constraints(
    finite_automaton: FiniteAutomaton,
    constraints_automaton: FiniteAutomaton,
    allow_start_to_end: bool = True,
) -> dict[int, set[int]]:
    """
    Вычисляет достижимость с регулярными ограничениями для нескольких стартовых вершин.

    Использует разреженные матрицы из sciPy для реализации алгоритма на основе multiple source BFS через линейную алгебру.

    Args:
        finite_automaton (FiniteAutomaton): Автомат, для которого вычисляется достижимость.
        constraints_automaton (FiniteAutomaton): Автомат с ограничениями, определяющий допустимые переходы.
        allow_start_to_end (bool, optional): Позволяет ли переход из стартовых состояний в конечные состояния. По умолчанию True.

    Returns:
        dict[int, set[int]]: Словарь, где ключи - стартовые состояния, а значения - множества достижимых из них состояний.
    """
    # Создаем словарь для хранения матриц переходов для каждого общего символа.
    transition_matrices = {
        symbol: block_diag(
            (
                constraints_automaton.transition_functions[symbol],
                finite_automaton.transition_functions[symbol],
            )
        )
        for symbol in finite_automaton.transition_functions.keys()
        & constraints_automaton.transition_functions.keys()
    }

    # Вычисляем общую высоту и ширину для матрицы переходов.
    matrix_height = len(constraints_automaton.state_index_map)
    matrix_width = matrix_height + len(finite_automaton.state_index_map)

    # Инициализируем словарь для хранения достижимых состояний для каждой стартовых вершины.
    reachable_states = {
        state.value: set() for state in finite_automaton.state_index_map
    }

    # Функция для диагонализации матрицы переходов.
    def diagonalize_transition_matrix(matrix):
        """
        Диагонализирует матрицу переходов, позволяя использовать линейную алгебру для обновления фронтира.

        Args:
            matrix (dok_matrix): Матрица переходов, которую нужно диагонализировать.

        Returns:
            dok_matrix: Диагонализированная матрица переходов.
        """
        result = dok_matrix(matrix.shape, dtype=bool)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[j, i]:
                    result[i] += matrix[j]
        return result

    # Функция для обновления фронтира.
    def update_frontier(frontier, transition_matrix):
        """
        Обновляет фронтир, используя диагонализированную матрицу переходов.

        Args:
            frontier (dok_matrix): Текущий фронтир, который нужно обновить.
            transition_matrix (dok_matrix): Матрица переходов, используемая для обновления фронтира.

        Returns:
            dok_matrix: Обновленный фронтир.
        """
        new_frontier = dok_matrix(frontier.shape, dtype=bool)
        for symbol in transition_matrices:
            new_frontier += diagonalize_transition_matrix(
                frontier @ transition_matrices[symbol]
            )
        return new_frontier

    # Для каждой стартовых вершины автомата с ограничениями.
    for start_state in finite_automaton.start_states:
        # Инициализируем фронтир для хранения достижимых состояний.
        frontier = dok_matrix((matrix_height, matrix_width), dtype=bool)
        for constraints_start_state in constraints_automaton.start_states:
            frontier[constraints_start_state, constraints_start_state] = True
        for i in range(matrix_height):
            frontier[i, start_state + matrix_height] = True

        # Проходим через все состояния автомата с ограничениями и автомата.
        for _ in range(matrix_height * len(finite_automaton.state_index_map)):
            frontier = update_frontier(frontier, transition_matrices)
            for i in range(matrix_height):
                if i in constraints_automaton.final_states and frontier[i, i]:
                    for j in range(len(finite_automaton.state_index_map)):
                        if (
                            j in finite_automaton.final_states
                            and frontier[i, j + matrix_height]
                        ):
                            if (
                                allow_start_to_end
                                or finite_automaton.state_index_map[start_state]
                                != finite_automaton.state_index_map[j]
                            ):
                                reachable_states[
                                    finite_automaton.state_index_map[start_state]
                                ].add(finite_automaton.state_index_map[j])

    return reachable_states
