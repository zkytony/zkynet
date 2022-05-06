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
    if node.parent_input_name is not None\
       and node.parent is not None:
        if isinstance(node, cg.FunctionNode):
            node_label = f"{node.function.name}→({node.value})"
        else:
            node_label = f"{node.name} ({node.value})"
    else:
        if isinstance(node, cg.FunctionNode):
            node_label = f"{node.function.name} ({node.value})"
        else:
            node_label = f"root ({node.value})"
    return node_label


def plot_cg(root, save_path="/tmp/cg"):
    """
    Visualize the computational graph headed by 'root'

    Args:
        root (cg.Node)
    """
    dot = graphviz.Digraph(comment=f"Computational Graph for {root.function.name}")

    # First get all the nodes and the edges.
    worklist = [root]
    seen = {root}
    while len(worklist) > 0:
        node = worklist.pop()
        _add_node(dot, node)
        for child in node.children:
            if child not in seen:
                dot.edge(child.id, node.id)
                worklist.append(child)
                seen.add(child)
    print(dot.source)
    dot.render(f"{save_path}.gv", view=True)
