"""
Visualize computational graph

Requires graphviz Python package: https://github.com/xflr6/graphviz.

    pip install graphviz
"""
from .framework import cg
import graphviz

def _add_node(dot, node):
    return dot.node(node.id, _node_label(node))

def _node_label(node):

    if isinstance(node, cg.OperatorNode):
        node_label = f"{node.operator.name}→({node.value})"
    else:
        if len(node.parents) > 0:
            node_label = f"{node.input.name} ({node.value})"
        else:
            node_label = f"root ({node.value})"
    return node_label


def plot_cg(root, save_path="/tmp/cg"):
    """
    Visualize the computational graph headed by 'root'

    Args:
        root (cg.Node)
    """
    dot = graphviz.Digraph(comment=f"Computational Graph")

    # First get all the nodes and the edges.
    worklist = [root]
    seen = {root}
    while len(worklist) > 0:
        node = worklist.pop()
        _add_node(dot, node)
        for child in node.children:
            if child not in seen:
                for parent in child.parents:
                    dot.edge(child.id, parent.id)
                worklist.append(child)
                seen.add(child)
    print(dot.source)
    dot.render(f"{save_path}.gv", view=True)
