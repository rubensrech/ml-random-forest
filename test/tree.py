from graphviz import Digraph

dot = Digraph()

class Node:
    pass

class AttrNode(Node):
    def __init__(self, attr):
        self.attr = attr
        self.children = {}
        dot.node(attr, attr, color='darkgreen')
    
    def getAttr(self):
        return self.attr
    
    def getChild(self, value):
        return self.children[value]
    
    def setChild(self, value, node):
        self.children[value] = node
        if isinstance(node,ClassNode):
            dot.edge(self.attr, ClassNode.NamePrefix + node.value, label=value)
        elif isinstance(node, AttrNode):
            dot.edge(self.attr, node.attr, label=value)

class ClassNode(Node):
    NamePrefix = 'class_'
    def __init__(self, value):
        self.value = value
        dot.node(ClassNode.NamePrefix + value, value, shape='rectangle', color='red')

def main():
    tree = AttrNode('weather')
    tree.setChild('sun', ClassNode('yes'))
    tree.setChild('rain', AttrNode('season'))
    dot.view()

if __name__ == "__main__":
    main()