from project.task3 import FiniteAutomaton, intersect_automata, transitive_closure


def reachability_with_constraints(fa: FiniteAutomaton, constraints_fa: FiniteAutomaton):
    """
    Вычисляет достижимость состояний в автомате с учетом ограничений.

    Args:
        fa (FiniteAutomaton): Автомат, для которого вычисляется достижимость.
        constraints_fa (FiniteAutomaton): Автомат, задающий ограничения.

    Returns:
        Dict[int, Set[int]]: Словарь, где ключи - это состояния в fa, а значения - множества состояний, достижимых из
        ключевого состояния с учетом ограничений.
    """
    intersection = intersect_automata(fa, constraints_fa, lbl=False)
    closure = transitive_closure(intersection)

    state_mapping = {v: i for i, v in fa.states_to_int.items()}
    constraints_len = len(constraints_fa.states_to_int)

    result = {start: set() for start in fa.start_states}

    for start, end in zip(*closure.nonzero()):
        if start in intersection.start_states and end in intersection.final_states:
            result[state_mapping[start // constraints_len]].add(
                state_mapping[end // constraints_len]
            )

    return result
