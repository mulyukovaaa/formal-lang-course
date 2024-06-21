from typing import Set, Tuple
from pyformlang.cfg import Terminal
from pyformlang.cfg.cfg import CFG
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cfg1 = cfg.eliminate_unit_productions().remove_useless_symbols()
    tmp = cfg1._get_productions_with_only_single_terminals()
    new_prod = cfg1._decompose_productions(tmp)
    return CFG(start_symbol=cfg1.start_symbol, productions=new_prod)


def cfpq_with_hellings(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    cfg = cfg_to_weak_normal_form(cfg)
    productions_epsilon = set()
    productions_single_terminal = {}
    productions_double_nonterminals = {}

    for production in cfg.productions:
        if len(production.body) == 0:
            productions_epsilon.add(production.head)
        if len(production.body) == 1 and isinstance(production.body[0], Terminal):
            productions_single_terminal.setdefault(production.head, set()).add(
                production.body[0]
            )
        elif len(production.body) == 2:
            productions_double_nonterminals.setdefault(production.head, set()).add(
                (production.body[0], production.body[1])
            )

    reachability_set = {
        (nonterminal, node, node)
        for nonterminal in productions_epsilon
        for node in graph.nodes
    }
    reachability_set |= {
        (nonterminal, node_v, node_u)
        for (node_v, node_u, tag) in graph.edges(data="label")
        for nonterminal, ls in productions_single_terminal.items()
        if Terminal(tag) in ls
    }

    work_set = reachability_set.copy()

    while len(work_set) > 0:
        nonterminal_i, node_v, node_u = work_set.pop()

        new_reachable = set()
        for nonterminal_j, node_v_, node_u_ in reachability_set:
            if node_v == node_u_:
                for (
                    nonterminal_k,
                    nonterminal_pairs,
                ) in productions_double_nonterminals.items():
                    if (nonterminal_j, nonterminal_i) in nonterminal_pairs and (
                        nonterminal_k,
                        node_v_,
                        node_u,
                    ) not in reachability_set:
                        work_set.add((nonterminal_k, node_v_, node_u))
                        new_reachable.add((nonterminal_k, node_v_, node_u))
            if node_v_ == node_u:
                for (
                    nonterminal_k,
                    nonterminal_pairs,
                ) in productions_double_nonterminals.items():
                    if (nonterminal_i, nonterminal_j) in nonterminal_pairs and (
                        nonterminal_k,
                        node_v,
                        node_u_,
                    ) not in reachability_set:
                        work_set.add((nonterminal_k, node_v, node_u_))
                        new_reachable.add((nonterminal_k, node_v, node_u_))
        reachability_set |= new_reachable

    return {
        (node_v, node_u)
        for nonterminal_i, node_v, node_u in reachability_set
        if node_v in start_nodes
        and node_u in final_nodes
        and nonterminal_i == cfg.start_symbol
    }
