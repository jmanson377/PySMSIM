import numpy as np

class SMSIM:
    def __init__(self, bounds, method='poly'):
        self.nvar = len(bounds[0])
        assert len(bounds[0]) == len(bounds[1])
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.method = method
        
    def initial(self):
        """Generate initial conditions for optimisation given instantiation"""
        raise NotImplementedError

    def optimise(self, func):
        """Closed loop optimisation of a given function, will only output optimisation result to user"""
        raise NotImplementedError

    def suggest(self):
        """Suggest conditions if user is running a batch optimisation"""
        raise NotImplementedError    

    def boundary_check(self):
        """Check for boundary violations"""
        raise NotImplementedError