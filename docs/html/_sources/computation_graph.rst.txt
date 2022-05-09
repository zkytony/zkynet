Computational Graph and Automatic Differentiation Framework
===========================================================


.. contents:: Table of Contents
    :depth: 3

The idea
--------

The idea of the computational graph framework breaks down
into two, interdependent parts:
(1) Function template specification;
(2) Computational graph grounding


Function template specification
-------------------------------

.. autoclass:: zkynet.framework.computation_graph.Function
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.Operator
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.Module
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.Input
   :members:


Computational graph grounding
-----------------------------

.. autoclass:: zkynet.framework.computation_graph.Node
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.InputNode
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.OperatorNode
   :members:
   :special-members:

.. autoclass:: zkynet.framework.computation_graph.ModuleGraph
   :members:
