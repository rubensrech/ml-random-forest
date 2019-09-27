from graphviz import Digraph

dot = Digraph()

class Node:
    uid = 0
    def __init__(self):
        self.id = "p" + str(Node.uid)
        Node.uid += 1

    def getID(self):
        return self.id

class AttrNode(Node):
    def __init__(self, attr):
        Node.__init__(self)
        self.attr = attr
        self.children = {}
        dot.node(self.getID(), attr, shape='rectangle', color='gold')
    
    def getAttr(self):
        return self.attr
    
    def getChild(self, value):
        return self.children[value]
    
    def setChild(self, value, node):
        self.children[value] = node
        if isinstance(node,ClassNode):
            dot.edge(self.getID(), node.getID(), label=value)
        elif isinstance(node, AttrNode):
            dot.edge(self.getID(), node.getID(), label=value)

class ClassNode(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value
        dot.node(self.getID(), value, color='darkgreen')

def main():
    tree = AttrNode('weather')
    tree.setChild('sun', ClassNode('yes'))
    tree.setChild('rain', AttrNode('season'))

    dot.view()

if __name__ == "__main__":
    main()