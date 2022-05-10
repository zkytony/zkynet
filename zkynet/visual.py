"""
Visualize computational graph

Requires graphviz Python package: https://github.com/xflr6/graphviz.

    pip install graphviz
"""
from .framework import cg
import graphviz
import signal
import cv2

def _add_node(dot, node):
    return dot.node(node.id, _node_label(node))

def _node_label(node):

    if isinstance(node, cg.OperatorNode):
        node_label = f"{node.operator.__class__.__name__}→({node.value})"
    else:
        if len(node.parents) > 0:
            node_label = f"{node.input.short_name} ({node.value})"
        else:
            node_label = f"root ({node.value})"
    return node_label


def plot_cg(root, save_path="/tmp/cg",
            quiet=True, view=True, wait=0, title=None):
    """
    Visualize the computational graph headed by 'root'

    Args:
        root (cg.Node)
        save_path (str) path to file to save. The resulting
            file name would be {save_path}.gv
        quiet (bool): True if suppress info printing
        view (bool): True if display the plot
        wait (number): seconds to wait until killing
            the visualization; 0 for infinity
        title (str): title to display on cv2 window
    """
    def _view_handler(signum, frame):
        raise TimeoutError("end of visualization!")

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
    if not quiet:
        print(dot.source)
    plot_path = f"{save_path}"
    dot.render(plot_path, format="png")
    plot_path = f"{save_path}.png"

    if view:
        img = cv2.imread(plot_path)
        if title is None:
            title = root.ref.functional_name
        cv2.imshow(title, img)
        cv2.waitKey(wait*1000)
        cv2.destroyAllWindows()
