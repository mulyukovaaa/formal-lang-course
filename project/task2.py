from typing import Set

from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Creates a minimal DFA by a given regular expression.

    Parameters:
    ------------
    regex: Regex
        Regular expression that will be used to build the DFA.
    Return:
    ------------
    DeterministicFiniteAutomaton
        Created minimized deterministic finite automaton.
    """
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> NondeterministicFiniteAutomaton:
    """Creates NFA from graph.

    Parameters:
    ------------
    graph: MultiDiGraph
        The graph for NFA.
    start_states: Set[int]
        Starting states. Maybe empty.
    final_states: Set[int]
        Final states. Maybe empty.
    Return:
    ------------
    NondeterministicFiniteAutomaton
        Created NFA.
    """

    nfa = NondeterministicFiniteAutomaton()

    start_nodes = start_nodes if start_nodes else set(graph.nodes())
    final_nodes = final_nodes if final_nodes else set(graph.nodes())

    for node in start_nodes:
        nfa.add_start_state(State(node))

    for final_node in final_nodes:
        nfa.add_final_state(State(final_node))

    for from_node, to_node, label in graph.edges(data="label"):
        nfa.add_transition(State(from_node), Symbol(label), State(to_node))

    return nfa
