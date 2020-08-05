import numpy as np
import unittest

class SMSIM:
    """ SMSIM Class for optimisation"""
    def __init__(self, bounds, x0, method='poly'):
        self.nvar = len(bounds[0])
        assert (len(bounds[0]) == len(bounds[1])), "Bounds are not of equal dimensions"
        assert (len(x0) == self.nvar), "Initial point dimension not equal to bound dimension" 
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.method = method
        self.x0 = x0
        self.X = None
        self.y = [[]]
        self.Xnext = None
        self.simplexN = 0
        self.covLimit = 0.5
        
    def __initial(self):
        """
        Generate initial conditions for optimisation given instantiation
        """
        delta = 0.05
        zero_delta = 0.00025
        x_initial = []
        x_initial.append(self.x0)
        for i in range(len(self.x0)):
            x = self.x0[:]
            if x[i] != 0:
                x[i] = (1+delta)*x[i]
            else:
                x[i] = zero_delta
            x_initial.append(x)

        return np.array(x_initial)

    def suggest(self, y=None):
        """
        Suggest conditions if user is running a batch optimisation
         Initial -> midpoint (don't request, just calc) -> reflection (request) -> optimum(request) -> calc next initial and loop back to midpoint (also check if finished)
        """
        if self.X == None:
            ncurrent = 0
        else:
            ncurrent = np.shape(self.X[-1])

        if ncurrent <= 0 and y == None: # Check if very first call
            self.Xnext = self.__initial()
            self.X = [self.Xnext]
            return self.Xnext
        elif ncurrent == (self.nvar + 1): # Midpoint calculation and reflection request
            self.y[-1].append(y)
            # Sort outputs to determine midpoint
            ind = np.argsort(y)
            mid = np.mean(self.X[-1][ind[:-1]])
            mid_p = np.mean(self.y[-1][ind[:-1]])
            
            raise NotImplementedError
        elif ncurrent == (self.nvar + 2): # Optimum calculation and request
            raise NotImplementedError
        elif ncurrent == (self.nvar + 3): # Calculate retention and move onto next 'simplex' iteration
            raise NotImplementedError
        raise NotImplementedError

    def optimise(self, func):
        """Closed loop optimisation of a given function, will only output optimisation result to user"""
        raise NotImplementedError       
    
    def __boundary_check(self):
        """Check for boundary violations"""
        raise NotImplementedError

    def __cov_calc(self, responses):
        """Calculate COV for termination criteria"""
        return 100 * np.std(responses) / np.mean(responses)

    def __rank(self, y):
        """Rank vertexes"""
        ind = np.argsort(y)
        raise NotImplementedError




class Tests(unittest.TestCase):
    def __init__(self):
        self.initialexpected = np.array([[1, 2], [1.05, 2], [1, 2.1]])
        smsim = SMSIM([[0,0],[3,3]], x0=[1,2])
        self.initialresult = smsim.suggest()

    def test_initial(self):
        assert(self.initialexpected == self.initialresult).all(), "Initial condition failure"
    

if __name__ == "__main__":
    test = Tests()
    test.test_initial()