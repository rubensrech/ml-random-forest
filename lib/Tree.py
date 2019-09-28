from graphviz import Digraph

class TreeNode:
    uid = 0
    
    def __init__(self, graph=None):
        self.id = "n" + str(TreeNode.uid)
        self.graph = graph
        TreeNode.uid += 1

    def getID(self):
        return self.id

class AttrNode(TreeNode):
    def __init__(self, attr, gain=None, graph=None):
        super().__init__(graph)
        self.attr = attr
        self.gain = gain
        self.children = {}
        if graph is not None:
            graph.node(self.getID(), self.__getGraphValue(), color='gold', shape='rectangle', fontsize='10.0')

    def __getGraphValue(self):
        if self.gain is None:
            return self.attr
        else:
            return self.attr + '\\nGain = ' + '{0:.3f}'.format(self.gain)

    def getChild(self, value):
        return self.children[value]
    
    def setChild(self, value, node):
        self.children[value] = node
        if self.graph is not None:
            if isinstance(node, ClassNode):
                self.graph.edge(self.getID(), node.getID(), label=value)
            elif isinstance(node, AttrNode):
                self.graph.edge(self.getID(), node.getID(), label=value)

class ClassNode(TreeNode):
    def __init__(self, value, instances=-1, graph=None):
        super().__init__(graph)
        self.value = value
        self.instances = instances
        if graph is not None:
            graph.node(self.getID(), self.__getGraphValue(), color='darkgreen')

    def __getGraphValue(self):
        if self.instances > -1:
            return self.value + ' (' + str(self.instances) + ')'
        else:
            return self.value