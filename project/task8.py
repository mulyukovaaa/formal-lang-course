import networkx as nx
import pyformlang
from pyformlang.cfg import Epsilon
from pyformlang.finite_automaton import Symbol
from pyformlang.regular_expression import Regex
from pyformlang.rsa import Box
from scipy.sparse import dok_matrix, eye
from project.task2 import graph_to_nfa
from project.task3 import (
    FiniteAutomaton,
    transitive_closure,
    intersect_automata,
    rsm_to_fa,
)


def cfpq_with_tensor(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.MultiDiGraph,
    final_nodes: set[int] = None,
    start_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_fa = rsm_to_fa(rsm)
    graph_fa = FiniteAutomaton(graph_to_nfa(graph, start_nodes, final_nodes))

    n = len(graph_fa.states_to_int)
    mat_idxs = rsm_fa.revert_mapping()
    graph_idxs = graph_fa.revert_mapping()

    for eps in rsm_fa.epsilons:
        graph_fa.matrix.setdefault(eps, eye(n, dtype=bool))

    len_closure = -1
    while True:
        closure = transitive_closure(intersect_automata(rsm_fa, graph_fa))
        closure = list(zip(*closure.nonzero()))

        if len_closure == len(closure):
            break
        len_closure = len(closure)

        for i, j in closure:
            frm, to = mat_idxs[i // n], mat_idxs[j // n]
            if frm in rsm_fa.start_states and to in rsm_fa.final_states:
                symbol = frm.value[0]
                graph_fa.matrix.setdefault(symbol, dok_matrix((n, n), dtype=bool))
                graph_fa.matrix[symbol][i % n, j % n] = True

    result = {
        (graph_idxs[i], graph_idxs[j])
        for _, matrix in graph_fa.matrix.items()
        for i, j in zip(*matrix.nonzero())
        if graph_idxs[i] in rsm_fa.start_states and graph_idxs[j] in rsm_fa.final_states
    }

    return result


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    boxes = {
        Box(
            Regex("$")
            .union(
                Regex(
                    " ".join(
                        "$" if isinstance(var, Epsilon) else var.value
                        for var in production.body
                    )
                )
            )
            .to_epsilon_nfa()
            .minimize(),
            Symbol(production.head),
        )
        for production in cfg.productions
    }

    return pyformlang.rsa.RecursiveAutomaton(
        set(production.head for production in cfg.productions), Symbol("S"), boxes
    )


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return pyformlang.rsa.RecursiveAutomaton.from_text(ebnf)
