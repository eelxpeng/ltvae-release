'''
Created on 20 Sep 2018

@author: Bryan
'''
from pyltm.model.node.continuous_belief_node import ContinuousBeliefNode
from pyltm.model.node.discrete_belief_node import DiscreteBeliefNode

class BifWriter(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def write(self, net, filename):
        with open(filename, "w") as writer:
            self.writeNetworkDeclaration(net, writer);
            self.writeVariables(net, writer);
            self.writeProbabilities(net, writer);
        
    def writeNetworkDeclaration(self, net, writer):
        writer.write('network "%s" {\n}\n' % net.getName())
        writer.write('\n')
    
    def writeVariables(self, net, writer):
        for node in net.nodes:
            if isinstance(node, ContinuousBeliefNode):
                for variable in node.variable.variables:
                    writer.write('variable "%s" {\n' % variable.name)
                    writer.write('\ttype continuous;\n')
                    writer.write('}\n\n')
            elif isinstance(node, DiscreteBeliefNode):
                writer.write('variable "%s" {\n' % node.variable.name)
                writer.write('\ttype discrete[%d] { ' % len(node.variable.states))
                for state in node.variable.states:
                    writer.write('"%s" ' % state)
                writer.write("};\n")
                writer.write('}\n\n')
            else:
                raise Exception("unknow belief node!")
    
    def writeProbabilities(self, net, writer):
        for node in net.nodes:
            parents = node.getDiscreteParentVariables()
            writer.write("probability (")
            writer.write(self.getNodeVariableName(node))
            if len(parents) > 0:
                writer.write(" | ")
                writer.write(", ".join([('"%s"' % v.name) for v in parents]))
            writer.write(') {\n')
            self.writeProbabilitiesWithStates(node, writer)
            writer.write('}\n\n')
            
    def getNodeVariableName(self, node):
        if isinstance(node, DiscreteBeliefNode):
            return ('"%s"' % node.variable.name)
        elif isinstance(node, ContinuousBeliefNode):
            return ", ".join([('"%s"' % v.name) for v in node.variable.variables])
        
    def writeProbabilitiesWithStates(self, node, writer):
        parents = node.getDiscreteParentVariables()
        node.potential, node.variable, parents
        if isinstance(node, DiscreteBeliefNode):
            # should only have 1 variable and 1 parent
            if len(parents)==0:
                writer.write("\ttable ");
                writer.write(" ".join([str(x) for x in node.potential.parameter.prob]))
                writer.write(";\n")
                return
            parent = parents[0]
            variables = [node.variable] + [parent]
            arr = node.potential.getCells(variables)
            for i in range(parent.getCardinality()):
                writer.write('\t("state%d") ' % i)
                writer.write(' '.join([str(x) for x in arr[:, i]]))
                writer.write(';\n')
        elif isinstance(node, ContinuousBeliefNode):
            # should only have 1 parent
            parent = parents[0]
            for i in range(node.potential.size):
                writer.write('\t("state%d") ' % i)
                writer.write(' '.join([str(x) for x in node.potential.get(i).mu]))
                writer.write(' ')
                writer.write(' '.join([str(x) for x in node.potential.get(i).covar.flatten()]))
                writer.write(';\n')
        else:
            raise Exception("unknow belief node!")