# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from math import exp
import random
import sys

class Game:
    def __init__(self, T=10):
        ###row and column indices together make the state. R is the reward table for states.
        R = np.zeros((10,10))
        
        R[5,5]= 1 #only 1 cell has +1 reward
        
        negRCells = [[3, 3], [4,5], [4,6], [5,6], [5,8], [6,8], [7,3],
                     [7,5],[7,6]]
        
        for cell in negRCells:
            R[cell[0], cell[1]] = -1
        
        Walls = np.zeros((10,10))
        for j in range(1,5):
            Walls[2,j] = 1
        for j in range(6,8):
            Walls[2,j] = 1
         
        for i in range(3,7):
            Walls[i,4] = 1
            
        self.R = R
        self.Walls = Walls
        self.beta = 0.9
        self.alpha = 0.01
        
        self.V = np.zeros((10,10))
        self.T = T  ##temperature
        self.Q = np.zeros((10,10,4))
        self.all_actions = [0, 1, 2, 3]
        self.all_action_letters = ['>', '<', '^', 'v']
        self.optimal_path = np.zeros((10,10))
        
        
    def V_func(self, i, j):
        self.V[i, j] = max(self.Q[i,j,0],self.Q[i,j,1], self.Q[i,j,2], self.Q[i,j,3])
        return self.V[i, j]
    
    def nextState(self, i, j, a):
        iprime = i
        jprime = j
        
        if a == 0:  #move right
            jprime+=1
        elif a==1:  #move left
            jprime-=1
        elif a==2:  #move down
            iprime+=1
        elif a == 3:  #move up
            iprime-=1
        else:
            print('error')
            
            
        #print(iprime, jprime)
        if iprime < 0 or jprime < 0 or iprime >=10 or jprime >=10 or self.Walls[iprime, jprime]==1:
            return i,j
        else:
            return iprime, jprime
    
    def updateQ(self, i, j, a):
        iprime, jprime = self.nextState(i, j, a)  
        ''' 
        Here the action outcome is deterministic, so no transition probability is needed
        '''
        self.Q[i, j, a] = self.R[i,j] + self.beta*self.V_func(iprime, jprime)  #
        if self.greedy:
            
            self.V[i, j]= self.V[i, j] + self.alpha*(self.R[i,j]+self.beta*self.V[iprime, jprime]-self.V[i,j])
        return iprime, jprime
        
    def prob_a_given_s(self,i,j,a):
        x = exp(self.Q[i,j, a]/self.T)
        y = x
        for a_other in self.all_actions:
            if a_other != a:   #we have already calculated this for one action,
                #by not calling it again, we are saving almost a 25% overhead in this function
                y+= exp(Q[i,j, a_other]/self.T) 
                
        return x/y
    
    def determine_next_state(self, i, j):
        all_probs = []  #to calculate the argmax, we do not need to calculate the denominator. We can
        #just return the action for which we get the highest numerator.
        
        for a in self.all_actions:
            all_probs.append(exp(self.Q[i,j, a]/self.T))
        all_probs = np.array(all_probs)
        denominator = np.sum(all_probs)
        weights = all_probs/denominator
        if not self.greedy: 
            return random.choices(self.all_actions, weights=weights, k=1)[0]
        
        else: #eps greedy; Qmax with prob 1-eps or any action with prob eps
            '''
            epsilon greedy implementation
            '''
            pivot = random.uniform(0.0, 1.0)
            if pivot >= self.eps:  ###prob 1 - eps
                return self.all_actions[np.argmax(np.array(all_probs))] #returning the action for which the prob
                                                            #is the highest
            else:  ###prob eps
                pivot = random.uniform(0.0, 1.0)
                if pivot <0.25:
                    return 0
                if pivot < 0.5:
                    return 1
                if pivot < 0.75:
                    return 2
                
                return 3
    
    
    
def QtoCSV(Q, filename):
    txt = "state (cell_i-cell_j), >, <, ^, v\n"
    for i in range(10):
        for j in range(10):
            line = "({}-{})".format(i,j)
            for k in range(4):
                line+=","+str(Q[i,j,k])
                
            txt+=line+"\n"
    
    with open(filename, 'w') as file:
        file.write(txt)
    return txt



def Eps_greedy(eps):
    print("Eps greedy method started...")
    print("Eps:", eps)
    game = Game()
    game.greedy = True
    game.eps = eps
    print("Reward table:")
    print(game.R)
    print("Positions of the walls:")
    print(game.Walls)
    
    i = 0
    j = 0
    _iter=0
    _iter2 = 0
    total_iter = 0
    while total_iter < 100000:
        _iter2 = 0
        
        while i!=5 or j!=5:
            #print("Iteration",_iter)
            a = game.determine_next_state(i, j)
            #print(game.all_action_letters[a])
            i, j = game.updateQ(i, j, a)
            #print(i, j)
            
            _iter2+=1
            total_iter+=1
        #goal state has been reached, reset state
        i = 0
        j = 0
        print("Trial:", _iter+1, "Iteration taken to converge:", _iter2)
        _iter+=1
        
    #print(game.Q)
    
    txt  = QtoCSV(game.Q, "Q-greedy-{}.csv".format(eps))
    print("Q table:")
    print(txt)


def Boltzman_exploration():
    print("Boltzman exploitation method started...")
    game = Game(10)
    game.greedy = False
    print("Reward table:")
    print(game.R)
    print("Positions of the walls:")
    print(game.Walls)
    
    i = 0
    j = 0
    _iter=0
    _iter2 = 0
    total_iter = 0
    while total_iter < 100000:
        _iter2 = 0
        
        while i!=5 or j!=5:
            #print("Iteration",_iter)
            a = game.determine_next_state(i, j)
            #print(game.all_action_letters[a])
            i, j = game.updateQ(i, j, a)
            #print(i, j)
            if _iter%10 == 0 and game.T*0.999 > 2:
                game.T = game.T*0.999  ###gradually the T will decrease
                #print("game.T:", game.T)
            
            _iter2+=1
            total_iter+=1
        #goal state has been reached, reset state
        i = 0
        j = 0
        print("Trial:", _iter+1, "Iteration taken to converge:", _iter2)
        _iter+=1
        
    #print(game.Q)
    
    txt  = QtoCSV(game.Q, "Q-boltzman.csv")
    print("Q table:")
    print(txt)
        
def main():
    np.set_printoptions(suppress=True)
    
    print("Check the output in q_learning-output.txt file")
    print("Program is running:")
    backup_stdout = sys.stdout
    backup_stderr = sys.stderr
    outfile = None
    try:
        outfile = open('q_learning-output.txt', 'w')
        sys.stdout = outfile
        sys.stderr= outfile
    except:
        print("Could not open q_learning-output.txt file")
    
    Eps_greedy(0.1)
    Eps_greedy(0.2)
    Eps_greedy(0.3)
    Boltzman_exploration()
    
    
    try:
        sys.stdout = backup_stdout
        sys.stderr = backup_stderr
        outfile.close()
    except:
        print("Could not close q_learning-output.txt file")
        
    print("Program is complete")
    print("Check the Q values in Q-boltzman.csv and Q-egreedy.csv files")
    
if __name__=="__main__":
    main()