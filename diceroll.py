
'''
Diceroll: A Python script to calculate and plot dice roll probabilities.
Updated 8-16-2022
By Erik Nykwest

This is a passion project to aid my game design hobby.
Probabilities are calculated DETERMINISTICALLY.
None of that lazy Monte Carlo non-sense.
The primary tool in this script is the Dice class.
For convenience, presets exist for Feng Shui style and FUDGE style dice.

ToDo
- Code Exploding Dice
- Addition of zero WAS broken, I think I fixed it, but I need to check it didn't break other things
- Added the use of negative multipliers, I don't promise it works.
- Change the .prob attribute to a @property
    - What about "Other Statistics"?

!!!WARNING!!!
- Changed to git style version control
- While python has infinite precision for integers, numpy does NOT.
    - Adding advantage multiple times can sometimes cause values to loop back to zero,
    - Explicitly setting the dtype to int64 helps, but doesn't solve this.
    - I don't think I care enough to to fix this, but an "infinite precision" mode might be nice.
'''

from itertools import product
import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt

try:
    from math import comb # require Python >= 3.8

except(ImportError):
    #print("No package comb in math")
    from math import factorial

    def comb(n,k):
        C = factorial(n)/(factorial(k)*factorial(n-k))
        return C


class Dice:
    """
    Use Dice(n) to create a fair n-sided dice.
    Adding dice means rolling both and adding the result.
    Multipling by a int "n" means to roll "n" of the same dice
    and add the result.
    """
    # core properties
    def __init__ (self, n = 1, advantage = 0):
        # Create with add_advantage if applicable
        if (type(advantage) == int) & (advantage != 0):
            # Creates dice with traditional advantage = X,
            # i.e. roll X times and take the highest number.
            m = advantage
            N = Dice(n)
            N2 = Dice(n)
            while m > 0:
                N = N.add_advantage(N2)
                m -= 1
                
            self.vals = N.vals 
            self.freq = N.freq 
            self.prob = N.prob
            self.total = N.total
            
            return
        
        elif (type(advantage) == list):
            # Used to create dice with asymmetric advantage.
            # for example for a d20 with advantage = [4]
            # you roll 1d20 and 1d4 and take the highest result.
            N = Dice(n)
            for m in advantage:
                M = Dice(m)
                N = N.add_advantage(M)
                
            self.vals = N.vals 
            self.freq = N.freq 
            self.prob = N.prob
            self.total = N.total
            
            return

        # Possible Dice Values / Faces
        self.vals = np.array(range(n), dtype='int64') +1
        
        # number of times each number / face appears on the dice
        count = [1]*n
        count = np.array(count, dtype='int64')
        self.freq = count
		
        # total number of sides / possibilities
        self.total = int(n)
		
        # map freq to prob
        self.prob = count.astype(float)/n

    
    # Methods
    def add_advantage(self, other):
            """
            Roll two dice and take the highest number
            as the final result.
            """
            # create a new die
            S = Dice()
            S.freq = dict()
    		
            # All possible combinations of dice rolls
            VF1 = [ [x,y] for (x,y) in zip(self.vals, self.freq)]
            VF2 = [ [x,y] for (x,y) in zip(other.vals, other.freq)]
            p = product(VF1, VF2) # or .items if dict
    
            # Consolidate duplicate values into 1 total
            for (n1, v1), (n2, v2) in p:
                # take higher of two rolls
                m = max(n1,n2)
                try:
                    S.freq[m] += v1*v2
                except(KeyError):
                    S.freq[m] = v1*v2
            
            # Convert dict to np.array
            S.freq = np.array(list(S.freq.items()))
            S.vals = S.freq[:, 0]
            S.freq = S.freq[:, 1] # this may have an issue
    
            # Total number of possible outcomes
            S.total = S.freq.sum()
    		
            # Calculate Probabilities
            S.prob = S.freq.astype(float)/S.total
            
            return S

        	
    # Adding dice means rolling both and adding the result
    def __add__(self, other):
        # create a new die
        S = Dice()
        S.freq = dict()
		
        # Enable integer support
        if type(other) == int:
            d1 = Dice(1)
            d1.vals[0] = other
            other = d1
        
        # All possible combinations of dice rolls
        VF1 = [ [x,y] for (x,y) in zip(self.vals, self.freq)]
        VF2 = [ [x,y] for (x,y) in zip(other.vals, other.freq)]
        p = product(VF1, VF2) # or .items if dict

        # Consolidate duplicate values into 1 total
        for (n1, v1), (n2, v2) in p:
            try:
                S.freq[n1+n2] += v1*v2
            except(KeyError):
                S.freq[n1+n2] = v1*v2
        
        # Convert dict to np.array
        S.freq = np.array(list(S.freq.items()))
        S.vals = S.freq[:, 0]
        S.freq = S.freq[:, 1] # this may have an issue

        # Total number of possible outcomes
        S.total = S.freq.sum()
		
        # Calculate Probabilities
        S.prob = S.freq.astype(float)/S.total
        
        return S
    
    # Substracting dice means rolling both and suntracting the result
    # of the SECOND dice from the result of the FIRST dice.
    def __sub__(self, other):
        # create a new die
        S = Dice()
        S.freq = dict()
        
        # Enable integer support
        if type(other) == int:
            d1 = Dice(1)
            d1.vals[0] = other
            other = d1
		
        # All possible combinations of dice rolls
        VF1 = [ [x,y] for (x,y) in zip(self.vals, self.freq)]
        # negative bc subtraction
        VF2 = [ [-x,y] for (x,y) in zip(other.vals, other.freq)]
        p = product(VF1, VF2) # or .items if dict

        # Consolidate duplicate values into 1 total
        for (n1, v1), (n2, v2) in p:
            try:
                S.freq[n1+n2] += v1*v2
            except(KeyError):
                S.freq[n1+n2] = v1*v2
        
        # Convert dict to np.array
        S.freq = np.array(list(S.freq.items()))
        S.vals = S.freq[:, 0]
        S.freq = S.freq[:, 1] # this may have an issue

        # Total number of possible outcomes
        S.total = S.freq.sum()
		
        # Calculate Probabilities
        S.prob = S.freq.astype(float)/S.total
        
        return S
    
    def __mul__(self, other):
        # Multiply Function: Dice * integer
        I = int(other)
        
        # Handle negetive numbers, if you're using this you're doing something weird.
        if I < 0:
            print('WARNING: Multiplying by a negative number only makes sense with subtraction!')
            R = self 
            R.vals *= -1
            I *= -1

        # Positive and zero multipliers
        if I == 0:
            R= Dice(1)
            R.vals[0] = 0
        else:    
            R = self
            while I > 1:
                R = R + self
                I -= 1
        
        return R

    def __rmul__(self, other):
        # Multiply Function: int * Dice
        R = self.__mul__(other)
        return R
    
    def roll(self, n =1):
        # roll the dice n times and return a list of the values
        R = random.choices(self.vals, weights= self.freq, k = n)
        return R
   
    
    # Plotting methods
    def plot_prob(self):
        #Plot a barchart
        fig = plt.bar(self.vals, self.prob) # if numpy array
        plt.ylabel("Probability")
        plt.xticks(self.vals)
        plt.xlabel("Roll = #")
        return fig

    def plot_gt(self, orequal=False):
        # Plot bar chart of the probability of rolling less than each number. 
        y = [ self.gt(n, orequal) for n in self.vals ]
        fig = plt.bar(self.vals, y) # if numpy array
        plt.ylabel("Probability")
        plt.xticks(self.vals)
        plt.xlabel("Roll > #")
        return fig

    def plot_lt(self, orequal=False):
        # Plot bar chart of the probability of rolling less than each number. 
        y = [ self.lt(n, orequal) for n in self.vals ]
        fig = plt.bar(self.vals, y) # if numpy array
        plt.ylabel("Probability")
        plt.xticks(self.vals)
        plt.xlabel("Roll < #")
        return fig
    
    def plot_bp(self, N, *values):
        """
        Plots the binimial probability.
        It plots the probability of rolling specific value(s)
        as a function of the number of times you roll that value.
        For example, plot_bp(3,20) would plot the probability of 
        rolling a twenty exactly 0, 1, 2, and 3 times
        if you rolled 3 dice.
        """
        
    
        # Plot the number of occurances (k) on the x axis
        Kvals = range(N+1)
        y = [ self.bp(N,k,values) for k in Kvals ]
        fig = plt.bar(Kvals, y) # if numpy array
        plt.ylabel("Probability")
        plt.xticks(Kvals)
        plt.xlabel("# of dice = {}".format(values))
        return fig
    
    
    # Probability Methods
    def p(self, *nums):
        '''Provide an int or list of ints.
        Returns the probability of rolling the given values.'''
        return self.prob[ np.isin(self.vals, nums) ].sum()

    def gt(self, n, orequal=False):
        if orequal:
            n-=1
        # probability of rolling > n, not including n.
        return self.prob[ self.vals > n ].sum()

    def lt(self,  n, orequal=False):
        if orequal:
            n+=1
        # probability of rolling < n, not including n.
        return self.prob[ self.vals < n ].sum()
        
    def bp(self, n, k, *nums, modify=None):
        '''
        Returns the probability of acheiving
        exactly k sucesses when rolling n dice.
        nums is a list of all values the count as success.
        
        Example 1:
        What is the probability of rolling EXACTLY 3 sixes with 5d6?
        _5d6 = 5*Dice(6)
        P = _5d6.bp(5,3,6)
        
        Example 2:
        What is the prob of of rolling at least 1 twenty with 2d20?
        (i.e. rolling with advantage)
        d20 = Dice(20)
        P = d20.bp(2,1,20) + d20.bp(2,2,20)
        
        # Alternatively
        P = d20.bp(2,[1,2],20)
        '''        
        
        # Modify functionality
        if modify == "or more":
            k = [ x for x in range(k,n+1) ]
            print(k)
        
        elif modify == "or less":
            k = k = [ x for x in range(0,k+1) ]
            print(k)
            
            
        # Probability of success
        p = self.p(nums)
        
        # If k is a number
        if (type(k) == int): 
            P = binomial_prob(n, k, p)
        
        # If k is a list
        else:
            P = 0
            for K in k:
                P += binomial_prob(n, K, p)
                
        return P
    
    
    # Other Statistics
    def avg(self):
        return np.dot(self.freq, self.vals)/self.total
    
    def _avg_of_sq(self):
        return np.dot(self.freq, self.vals**2)/self.total
    
    def _sq_avg(self):
        return (np.dot(self.freq, self.vals)/self.total)**2
    
    def var(self):
        return self._avg_of_sq() - self._sq_avg()
    
    def std(self):
        return sqrt(self.var())
        
    

### Functions
def binomial_prob(n,k,p):
    """
    Calculates the probability P that
    exactly k events with probability P
    will occur in n rolls.
  
    EX: What is the probability of rolling exactly 3 sixes with 6 dice?
        d6 = Dice(6)
        P6= d6.p(6) # Ths probability of rolling a six is 1/6
        binomial_prob(6,3,P6) # or binomial_prob(6,3,1/6)
    """
    q = 1-p
    P = comb(n,k)*(p**k)*(q**(n-k))
    
    return P


# Presets
#Still need to code exploding dice
FS = Dice(6)-Dice(6)

dF = Dice(3)
dF -= 2
Fudge = 4*dF


#%% Examples
if __name__ == "__main__":
    d6 = Dice(6) # create a traditional 6-sided die
    a2d6 = 2*d6 # figure out the probabilites for rolling two d6 and adding the results together
    a3d6 = 3*d6 # same as above but for 3d6
    print(a3d6.freq)
    print(a3d6.prob)
    a3d6.plot_prob()
    plt.show()
    
    print(d6.avg())
    print(d6.std())


