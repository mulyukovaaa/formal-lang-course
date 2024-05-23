from copy import deepcopy
import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from project.task8 import cfg_to_rsm


def cfpq_with_gll(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if isinstance(rsm, CFG):
        rsm = cfg_to_rsm(rsm)

    start_nodes = start_nodes or set(graph.nodes)
    final_nodes = final_nodes or set(graph.nodes)
    initial_label = rsm.initial_label.value or "S"
    start_states = {(initial_label, node) for node in start_nodes}
    stack_graph = {state: set() for state in start_states}
    visited = {
        (
            state[1],
            (initial_label, rsm.boxes[rsm.initial_label].dfa.start_state.value),
            state,
        )
        for state in start_states
    }
    queue = deepcopy(visited)

    def add(graph_node, rsm_state, stack_state):
        state = (graph_node, rsm_state, stack_state)
        if state not in visited:
            queue.add(state)
            visited.add(state)

    pop = {}
    result = set()

    while queue:
        graph_node, rsm_state, stack_state = queue.pop()

        if rsm_state[1] in rsm.boxes[rsm_state[0]].dfa.final_states:
            if stack_state in start_states and graph_node in final_nodes:
                result.add((stack_state[1], graph_node))
            pop.setdefault(stack_state, set()).add(graph_node)
            [
                add(graph_node, rsm_state_, stack_state_)
                for stack_state_, rsm_state_ in stack_graph.get(stack_state, set())
            ]

        next_states = {
            label: set() for _, _, label in graph.edges(graph_node, data="label")
        }
        [
            next_states[label].add(u)
            for _, u, label in graph.edges(graph_node, data="label")
        ]

        dfa_transitions = rsm.boxes[rsm_state[0]].dfa.to_dict().get(rsm_state[1], {})
        for symbol, to_state in dfa_transitions.items():
            if symbol not in rsm.labels:
                if symbol.value in next_states:
                    [
                        add(next_node, (rsm_state[0], to_state.value), stack_state)
                        for next_node in next_states[symbol.value]
                    ]

            else:
                stack_state_new = (symbol.value, graph_node)
                if stack_state_new in pop:
                    [
                        add(next_node, (rsm_state[0], to_state.value), stack_state)
                        for next_node in pop[stack_state_new]
                    ]

                stack_graph.setdefault(stack_state_new, set()).add(
                    (stack_state, (rsm_state[0], to_state.value))
                )
                add(
                    graph_node,
                    (symbol.value, rsm.boxes[symbol].dfa.start_state.value),
                    stack_state_new,
                )

    return result