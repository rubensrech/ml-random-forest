from graphviz import Digraph

dot = Digraph(edge_attr={'fontsize':'10.0'})

class Node:
    uid = 0
    def __init__(self):
        self.id = "p" + str(Node.uid)
        Node.uid += 1

    def getID(self):
        return self.id

class AttrNode(Node):
    def __init__(self, attr, gain=None):
        super(AttrNode, self).__init__()
        self.attr = attr
        self.gain = gain
        self.children = {}

        dot.node(self.getID(), self.__getGraphValue(), color='gold', shape='rectangle', fontsize='10.0')

    def __getGraphValue(self):
        if self.gain is None:
            return self.attr
        else:
            return self.attr + '\\nGain = ' + '{0:.3f}'.format(self.gain)
    
    def getAttr(self):
        return self.attr
    
    def getGain(self):
        return self.gain

    def getChild(self, value):
        return self.children[value]
    
    def setChild(self, value, node):
        self.children[value] = node
        if isinstance(node,ClassNode):
            dot.edge(self.getID(), node.getID(), label=value)
        elif isinstance(node, AttrNode):
            dot.edge(self.getID(), node.getID(), label=value)


class ClassNode(Node):
    def __init__(self, value, instances=-1):
        super(ClassNode, self).__init__()
        self.value = value
        self.instances = instances

        dot.node(self.getID(), self.__getGraphValue(), color='darkgreen')

    def __getGraphValue(self):
        if self.instances > -1:
            return self.value + ' (' + str(self.instances) + ')'
        else:
            return self.value

    def getValue(self):
        return self.value

def main():
    tree = AttrNode('weather', 0.127)
    tree.setChild('sun', ClassNode('yes'))
    tree.setChild('rain', AttrNode('season'))
    dot.view()

if __name__ == "__main__":
    main()