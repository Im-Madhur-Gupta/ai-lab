# Madhur Gupta
# 202051112

import numpy as np
import matplotlib.pyplot as plt

class NODE:
    def __init__(self, reward, correctProbability):
        self.reward = reward
        self.probab = correctProbability
        self.next = None

class AGENT:
    def __init__(self):
        self.N = 10
        self.theta = 0.0001
        self.gamma = 0.9
        self.ACTIONS = ["CONTINUE", "QUIT"]
        
        REWARDS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
        probab_answering_correctly = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        START_NODE = NODE(-1,0)
        TEMP = START_NODE
        
        for i in range(self.N):  
            TEMP.next = NODE(REWARDS[i],probab_answering_correctly[i])
            TEMP = TEMP.next
        
        self.START_STATE = START_NODE.next        


class MDP_SOLUTION:
    def __init__(self):
        self.agent = AGENT()
        self.val_func = {s: 0 for s in range(self.agent.N)}
        self.ITERATIONS = 0
        self.TIMES_ENTERED = {s: 0 for s in range(self.agent.N)}
        self.quit = False
        self.PLOT_STATES = [x for x in range(self.agent.N)]
    
    def helper(self, state, iteration):
        if(state == None): return 0
        self.TIMES_ENTERED[iteration] += 1
        OLD_VALUE = self.val_func[iteration]
        current_reward = 0
        if iteration == 0:
            quit_reward = 0
        else:
            quit_reward = self.val_func[iteration-1]
        
        ans = np.random.rand()
        if ans <= state.probab:
            current_reward = state.probab * (state.reward + (self.agent.gamma * self.helper(state.next, iteration+1)))
            self.val_func[iteration] = (self.val_func[iteration] * self.TIMES_ENTERED[iteration] + current_reward)/(self.TIMES_ENTERED[iteration]+1)
            
            if(abs(self.val_func[iteration] - OLD_VALUE) < self.agent.theta):
                self.quit = True

        return max(quit_reward, current_reward)
    
    def solver(self):
        while self.quit == False:
            self.ITERATIONS += 1
            HEAD = self.agent.START_STATE
            self.helper(HEAD, 0)
        
        print("Total Iterations: ", self.ITERATIONS)
        print("VALUE FUNCTION:")
        print(self.val_func)
        
        for i in range(self.agent.N):
            self.TIMES_ENTERED[i] = (self.TIMES_ENTERED[i] / self.ITERATIONS) * 100
        
        print("TIMES ENTERED:")
        print(self.TIMES_ENTERED)
        
        EXPECTATION = 0
        for i in range(self.agent.N):
            EXPECTATION = EXPECTATION + ((self.TIMES_ENTERED[i]/100) * self.val_func[i])
        
        print("EXPECTED reward: ", EXPECTATION)
        plt.bar(self.PLOT_STATES, self.val_func.values())
        plt.xlabel('States')
        plt.ylabel('Value Function')
        plt.title('Maxmimum Reward for each state')
        plt.show()
        
        plt.bar(self.PLOT_STATES, self.TIMES_ENTERED.values())
        plt.xlabel('States')
        plt.ylabel('Times Entered')
        plt.title('Number of times agent went into a particular state')
        plt.show()


MDP_SOLUTION().solver()