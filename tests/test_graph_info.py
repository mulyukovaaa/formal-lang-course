from project.graph_info import *
from pathlib import Path
import filecmp
import os


class TestInfo:
    def test_get_graph_info_graph_dataset(self):
        actual_graph = load_graph("skos")
        assert actual_graph.number_of_nodes() == 144
        assert actual_graph.number_of_edges() == 252

        actual_graph = load_graph("bzip")
        assert 632 == actual_graph.number_of_nodes()
        assert 556 == actual_graph.number_of_edges()

    def test_get_graph_info_nonempty_graph(self):
        graph = MultiDiGraph()
        for i in range(10):
            graph.add_node(i)

        graph.add_edge(1, 3, label="a")
        graph.add_edge(2, 3, label="b")
        graph.add_edge(4, 1, label="a")
        graph.add_edge(1, 4, label="c")
        graph.add_edge(1, 2, label="d")
        graph.add_edge(5, 6, label="e")
        graph.add_edge(4, 5, label="f")
        graph.add_edge(6, 7, label="g")
        graph.add_edge(7, 8, label="i")
        graph.add_edge(9, 10, label="k")
        graph.add_edge(8, 10, label="l")

        expected_graph_info = [
            11,
            11,
            ("a", "b", "c", "d", "e", "f", "g", "i", "k", "l"),
        ]

        actual_graph_info = get_graph_info(graph)

        assert actual_graph_info[0] == expected_graph_info[0]
        assert actual_graph_info[1] == expected_graph_info[1]
        assert set(actual_graph_info[2]) == set(expected_graph_info[2])

    def test_create_labeled_two_cycle_graph(self):
        graph = create_two_cycle_labeled_graph(2, 2, ("a", "b"))
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 6

    def test_save_graph_as_dot_file(self):
        first_cycle_nodes = 2
        second_cycle_nodes = 2
        labels = ("a", "b")
        path = Path("graph.dot")
        path_actual = Path("graph_compare.dot")

        graph = create_two_cycle_labeled_graph(
            first_cycle_nodes, second_cycle_nodes, labels
        )
        write_to_dot(graph, path)

        graph_actual = cfpq_data.labeled_two_cycles_graph(
            first_cycle_nodes, second_cycle_nodes, labels=labels
        )
        write_to_dot(graph_actual, path_actual)

        assert filecmp.cmp(path, path_actual)

        os.remove(path)
        os.remove(path_actual)
