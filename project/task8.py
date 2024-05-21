from project.task2 import graph_to_nfa
from pyformlang.rsa import RecursiveAutomaton, Box
from pyformlang.cfg import CFG

import networkx as nx
from itertools import *
from scipy.sparse import *

from project.task3 import FiniteAutomaton, rsm_to_fa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def cfpq_with_tensor(
    rsm: RecursiveAutomaton,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if isinstance(rsm, CFG):
        rsm = cfg_to_rsm(rsm)

    start_nodes = start_nodes or graph.nodes
    final_nodes = final_nodes or graph.nodes

    fa_rsm = rsm_to_fa(rsm)
    num_rsm_states = len(fa_rsm.states)
    fa_graph = FiniteAutomaton(graph_to_nfa(graph, start_nodes, final_nodes))
    state_mapping = {
        i: state for i, state in enumerate(product(fa_graph.states, fa_rsm.states))
    }

    num_graph_states = len(fa_graph.states)

    previous_non_zeros = 0
    while True:
        num_states = num_rsm_states * num_graph_states
        common_symbols = fa_rsm.matrix.keys() & fa_graph.matrix.keys()
        if common_symbols:
            matrix_sum = sum(
                kron(fa_graph.matrix[symbol], fa_rsm.matrix[symbol])
                for symbol in common_symbols
            )
        else:
            matrix_sum = dok_matrix((num_states, num_states), dtype=bool)

        matrix_sum += eye(num_states, dtype=bool)

        for _ in range(num_states):
            matrix_sum += matrix_sum @ matrix_sum

        current_non_zeros = matrix_sum.count_nonzero()
        if current_non_zeros <= previous_non_zeros:
            break
        else:
            previous_non_zeros = current_non_zeros

        for from_idx, to_idx in zip(*matrix_sum.nonzero()):
            from_state = state_mapping[from_idx]
            to_state = state_mapping[to_idx]
            from_rsm_state = from_state[1]
            to_rsm_state = to_state[1]
            if (
                from_rsm_state in fa_rsm.start_states
                and to_rsm_state in fa_rsm.final_states
            ):
                symbol = from_rsm_state[0]
                graph_from = fa_graph.states_to_int[from_state[0]]
                graph_to = fa_graph.states_to_int[to_state[0]]
                fa_graph.matrix.setdefault(
                    symbol, dok_matrix((num_graph_states, num_graph_states), dtype=bool)
                )[graph_from, graph_to] = True

    initial_symbol = rsm.initial_label.value
    if initial_symbol not in fa_graph.matrix:
        return set()

    result = set()
    for graph_from_state, graph_to_state in product(start_nodes, final_nodes):
        graph_from = fa_graph.states_to_int[graph_from_state]
        graph_to = fa_graph.states_to_int[graph_to_state]
        if fa_graph.matrix[initial_symbol][graph_from, graph_to]:
            result.add((graph_from_state, graph_to_state))

    return result