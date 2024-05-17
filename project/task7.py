import networkx as nx
import pyformlang
from typing import Set
from pyformlang.cfg import Terminal
from scipy.sparse import lil_matrix
from project.task6 import cfg_to_weak_normal_form


def cfpq_with_matrix(
    grammar: pyformlang.cfg.CFG,
    graph_data: nx.DiGraph,
    start_set: Set[int] = None,
    final_set: Set[int] = None,
) -> set[tuple[int, int]]:
    grammar = cfg_to_weak_normal_form(grammar)
    nonterminals_set = {rule.head for rule in grammar.productions}
    var_indices = {var: idx for idx, var in enumerate(nonterminals_set)}
    adj_matrices_dict = {
        variable: lil_matrix(
            (graph_data.number_of_nodes(), graph_data.number_of_nodes()), dtype=bool
        )
        for variable in nonterminals_set
    }

    for rule in grammar.productions:
        for edge_data in graph_data.edges(data=True):
            body_is_terminal = len(rule.body) == 1 and isinstance(
                rule.body[0], Terminal
            )
            edge_label_matches = (
                str(edge_data[2].get("label", "")) == str(rule.body[0])
                if body_is_terminal
                else False
            )
            if body_is_terminal and edge_label_matches:
                adj_matrices_dict[rule.head][edge_data[0], edge_data[1]] = True

    while True:
        changes_occurred = False

        for rule in grammar.productions:
            if len(rule.body) == 2:
                matrix_a, matrix_b = rule.body
                if matrix_a in var_indices and matrix_b in var_indices:
                    before_change = adj_matrices_dict[rule.head].nnz
                    adj_matrices_dict[rule.head] += (
                        adj_matrices_dict[matrix_a] * adj_matrices_dict[matrix_b]
                    )
                    after_change = adj_matrices_dict[rule.head].nnz
                    if before_change != after_change:
                        changes_occurred = True

        if not changes_occurred:
            break

    result_set = set()
    for variable, matrix_data in adj_matrices_dict.items():
        if variable == grammar.start_symbol:
            matrix_data = matrix_data.tocoo()
            for i, j in zip(matrix_data.row, matrix_data.col):
                if (start_set is None or i in start_set) and (
                    final_set is None or j in final_set
                ):
                    result_set.add((i, j))
    return result_set
