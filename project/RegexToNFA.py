from networkx import MultiDiGraph
from pyformlang.regular_expression import *
from pyformlang.finite_automaton import *


def create_minimal_dfa(regex: Regex) -> DeterministicFiniteAutomaton:
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
    return regex.to_epsilon_nfa().minimize()


# def create_nfa(
#         graph: MultiDiGraph,
#         start_vertices: set[int],
#         end_vertices: set[int]
# ) -> NondeterministicFiniteAutomaton:
#
#     nfa = NondeterministicFiniteAutomaton()
#
#     start_vertices = start_vertices if start_vertices else set(graph.nodes())
#     end_vertices = end_vertices if end_vertices else set(graph.nodes())
