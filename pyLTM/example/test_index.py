'''
Created on 14 Feb 2018

@author: Bryan
'''
import numpy as np
def getPermAxes(variables, subVars):
            index = list(range(len(variables)))
            transformAxes = []
            reverseAxes = [0]*len(variables)
            for i in range(len(subVars)):
                j = variables.index(subVars[i])
                transformAxes.append(j)
                index.pop(j)
                reverseAxes[j] = i
            for i in index:
                transformAxes.append(i)
                reverseAxes[i] = len(transformAxes)-1
            return transformAxes, reverseAxes
        
variables = [0,1,2,3]
subVars = [3,1]
transformAxes, reverseAxes = getPermAxes(variables, subVars)
transformed = np.array(variables)[transformAxes] 
print(transformed)
print(transformed[reverseAxes])
