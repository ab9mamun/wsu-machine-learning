# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np

class Game:
    def __init__(self):
        ###row and column indices together make the state. R is the reward table for states.
        R = np.array([[0 for i in range(10)] for j in range(10)])
        
        R[5,5]= 1 #only 1 cell has +1 reward
        
        negRCells = [[3, 3], [4,5], [4,6], [5,6], [5,8], [6,8], [7,3],
                     [7,5],[7,6]]
        
        for cell in negRCells:
            R[cell[0], cell[1]] = -1
        
        Walls = np.array([[0 for i in range(10)] for j in range(10)])
        for j in range(1,5):
            Walls[2,j] = 1
        for j in range(6,8):
            Walls[2,j] = 1
         
        for i in range(3,7):
            Walls[i,4] = 1
            
        self.R = R
        self.Walls = Walls
        self.beta = 0.9
        self.V = np.array([[0 for i in range(10)] for j in range(10)])
    def V_func(self, s_prime):
        return self.V[s_prime[0], s_prime[1]]
    
    def nextState(self, i, j, a):
        iprime = i
        jprime = j
        
        if a == '>':  #move right
            jprime+=1
        elif a=='<':  #move left
            jprime-=1
        elif a=='v':  #move down
            iprime+=1
        elif a == '^':  #move up
            iprime-=1
            
            
        
        if iprime < 0 or jprime < 0 or iprime >=10 or jprime >=10 or self.Walls[iprime, jprime]==1:
            return [i,j]
        else:
            return [iprime, jprime]
    
    def updateQ(self, i, j, a):
        s_prime = self.nextState(i, j, a)  
        ''' 
        Here the action outcome is deterministic, so no transition probability is needed
        '''
        self.Q[i, j, a] = self.R[i,j] + self.beta*self.V_func(s_prime)  #
        
def main():
    game = Game()
    print(game.R)
    print(game.Walls)
    
    
    
    
if __name__=="__main__":
    main()