from networkx import MultiDiGraph, drawing
import cfpq_data
from pathlib import Path


def get_graph_info(graph: MultiDiGraph) -> tuple[int, int, set]:
    """Get graph info

    Parameters:
    ----------
    graph : MultiDiGraph
        The graph from which to get the info.

    Returns:
    -------
    graph_info : tuple[int, int, set]
        The graph info.
    """

    labels = set([b for _, _, _, b in graph.edges(data="label", keys=True)])
    return graph.number_of_nodes(), graph.number_of_edges(), labels


def load_graph(graph_name: str) -> MultiDiGraph:
    """Load graph from dataset csv

    Parameters:
    ----------
    graph_name (str):
        The name of the graph dataset to load.

    Examples
    --------
    >>> import
    >>> graph = load_graph('example')

    Returns:
    ----------
    nx.MultiDiGraph
    """

    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    return graph


def create_two_cycle_labeled_graph(
    first_cycle_nodes: int, second_cycle_nodes: int, labels: tuple[str, str]
) -> MultiDiGraph:
    """Create labeled two cycles graph

    Parameters:
    ----------
    first_cycle_nodes: int
        The number of nodes in the first cycle.
    second_cycle_nodes: int
        The number of nodes in the second cycle.
    labels: tuple[str, str]
        Labels for graph.

    Returns:
    ---------
    graph: MultiDiGraph
        Returns a labeled graph with two cycles.
    """
    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels=labels
    )
    return graph


def write_to_dot(graph: MultiDiGraph, path: Path) -> None:
    """Write graph to dot file.

    Parameters:
    ----------
    graph: MultiDiGraph

    path: Path
    """
    data = drawing.nx_pydot.to_pydot(graph)
    data.write_raw(path)
