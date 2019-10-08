from graphviz import Digraph

class TreeNode:
    uid = 0
    
    def __init__(self, graph=None):
        self.id = "n" + str(TreeNode.uid)
        self.graph = graph
        TreeNode.uid += 1

    def getID(self):
        return self.id

    @staticmethod
    def isClassNode(node):
        return isinstance(node, ClassNode)

    @staticmethod
    def isAttrNode(node):
        return isinstance(node, AttrNode)

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
        # Might not have child 'value'
        return self.children[value]
    
    def setChild(self, value, node):
        self.children[value] = node
        if self.graph is not None:
            self.graph.edge(self.getID(), node.getID(), label=value)
    
    def isNumericalAttr(self):
        return isinstance(self, NumAttrNode)

    def countInstances(self):
        count = 0
        for v in self.children.keys():
            count += self.getChild(v).countInstances()
        return count

class NumAttrNode(AttrNode):
    def __init__(self, attr, gain=None, graph=None, cutoff=None):
        self.cutoff = cutoff
        super().__init__(attr, gain, graph)

    def setCutOff(self, cutoff):
        self.cutoff = cutoff

    def setLeftChild(self, node):
        if self.cutoff is None: return 
        self.children['left'] = node
        if self.graph is not None:
            self.graph.edge(self.getID(), node.getID(), label='<= {0:.2f}'.format(self.cutoff))

    def setRightChild(self, node):
        if self.cutoff is None: return 
        self.children['right'] = node
        if self.graph is not None:
            self.graph.edge(self.getID(), node.getID(), label='> {0:.2f}'.format(self.cutoff))

    def getChild(self, value):
        if self.cutoff is None: return
        if value <= self.cutoff:
            return self.children['left']
        else:
            return self.children['right']

    def countInstances(self):
        return self.children['left'].countInstances() + self.children['right'].countInstances()


class ClassNode(TreeNode):
    def __init__(self, value, instances=-1, graph=None):
        super().__init__(graph)
        self.value = value
        self.instances = instances
        if graph is not None:
            graph.node(self.getID(), self.__getGraphValue(), color='darkgreen')

    def __getGraphValue(self):
        if self.instances > -1:
            return "%s (%d)" % (self.value, self.instances)
        else:
            return self.value

    def countInstances(self):
        return self.instances