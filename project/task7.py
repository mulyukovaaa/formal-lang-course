import copy
from typing import Set
import networkx as nx
from pyformlang.cfg import CFG, Terminal
from scipy.sparse import dok_matrix
from project.task6 import cfg_to_weak_normal_form

def cfpq_with_matrix(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    cfg = cfg_to_weak_normal_form(cfg)

    matrix_dict = {}
    epsilon_set = set()
    terminal_dict = {}
    nonterminal_dict = {}

    for production in cfg.productions:
        if len(production.body) == 0:
            epsilon_set.add(production.head.to_text())
        if len(production.body) == 1 and isinstance(production.body[0], Terminal):
            terminal_dict.setdefault(production.body[0].to_text(), set()).add(production.head.to_text())
        matrix_dict[production.head.to_text()] = dok_matrix(
            (graph.number_of_nodes(), graph.number_of_nodes()), dtype=bool
        )
        if len(production.body) == 2:
            nonterminal_dict.setdefault(production.head.to_text(), set()).add(
                (production.body[0].to_text(), production.body[1].to_text())
            )

    for start_node, end_node, label in graph.edges(data="label"):
        if label in terminal_dict:
            for terminal_symbol in terminal_dict[label]:
                matrix_dict[terminal_symbol][start_node, end_node] = True

    for epsilon_symbol in epsilon_set:
        matrix_dict[epsilon_symbol].setdiag(True)

    new_matrix_dict = copy.deepcopy(matrix_dict)
    [m.clear() for m in new_matrix_dict.values()]

    for _ in range(graph.number_of_nodes() ** 2):
        for nonterminal, nonterminal_pairs in nonterminal_dict.items():
            for left_nonterminal, right_nonterminal in nonterminal_pairs:
                new_matrix_dict[nonterminal] += matrix_dict[left_nonterminal] @ matrix_dict[right_nonterminal]
        for nonterminal, m in new_matrix_dict.items():
            matrix_dict[nonterminal] += m

    start_symbol = cfg.start_symbol.to_text()
    ns, ms = matrix_dict[start_symbol].nonzero()
    return {(n, m) for n, m in zip(ns, ms) if n in start_nodes and m in final_nodes}