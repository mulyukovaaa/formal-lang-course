from scipy.sparse import block_diag, dok_matrix
from project.task3 import FiniteAutomaton


def diagonalise(m):
    res = dok_matrix(m.shape, dtype=bool)
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if m[j, i]:
                res[i] += m[j]
    return res


def reachability_with_constraints(
    fa: FiniteAutomaton, constraints_fa: FiniteAutomaton
) -> dict[int, set[int]]:
    matrices = {
        line: block_diag((constraints_fa.matrix[line], fa.matrix[line]))
        for line in fa.matrix.keys() & constraints_fa.matrix.keys()
    }

    result = {start: set() for start in fa.states}

    fa_start_states = {fa.states_to_int[i] for i in fa.start_states}
    fa_final_states = {fa.states_to_int[i] for i in fa.final_states}
    con_start_states = {
        constraints_fa.states_to_int[i] for i in constraints_fa.start_states
    }
    con_final_states = {
        constraints_fa.states_to_int[i] for i in constraints_fa.final_states
    }

    len_con = len(constraints_fa.states)
    len_fa = len(fa.states)

    for state in fa_start_states:
        front = dok_matrix((len_con, len_con + len_fa), dtype=bool)
        front[list(con_start_states), list(con_start_states)] = True
        front[:, state + len_con] = True

        if state in fa_final_states and con_start_states & con_final_states:
            result[fa.states[state]].add(fa.states[state])

        for _ in range(len_con * len_fa):
            new_front = dok_matrix((len_con, len_con + len_fa), dtype=bool)
            for line in fa.matrix.keys() & constraints_fa.matrix.keys():
                new_front += diagonalise(front @ matrices[line])
            front = new_front
            for i in con_final_states:
                if front[i, i]:
                    result[fa.states[state]].update(
                        fa.states[j] for j in fa_final_states if front[i, len_con + j]
                    )
    return result
