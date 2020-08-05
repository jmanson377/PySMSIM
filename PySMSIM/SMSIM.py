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
        
    def initial(self):
        """Generate initial conditions for optimisation given instantiation"""
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

        return x_initial

    def optimise(self, func):
        """Closed loop optimisation of a given function, will only output optimisation result to user"""
        raise NotImplementedError

    def suggest(self):
        """Suggest conditions if user is running a batch optimisation"""
        raise NotImplementedError    

    def boundary_check(self):
        """Check for boundary violations"""
        raise NotImplementedError

class Tests(unittest.TestCase):
    def __init__(self):
        self.initialexpected = [[1, 2], [1.05, 2], [1, 2.1]]
        smsim = SMSIM([[0,0],[3,3]], x0=[1,2])
        self.initialresult = smsim.initial()

    def test_initial(self):
        self.assertCountEqual(self.initialexpected, self.initialresult)
    

if __name__ == "__main__":
    test = Tests()
    test.test_initial()