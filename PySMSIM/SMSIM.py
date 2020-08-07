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
        self.y = []
        self.Xnext = None
        self.simplexN = 0
        self.covLimit = 0.5
        self.Ya = None
        self.ind = None
        self.mid = None
        self.mid_p = None
        self.YoptMin = -1
        self.YoptMax = 3
        self.YoptMargin = 0.3

        
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

    def __boundary_check(self, suggestion, mid, worst):
        """Check for boundary violations. N.B. I think this can be optimised"""
        Ya = None
        index_lb = suggestion < self.lb
        index_ub = suggestion > self.ub
        modified = suggestion
        while np.any(index_lb) and np.any(index_ub):
            if np.any(index_lb): # lower bound violation
                Ya = 1 + ((self.lb[index_lb] - mid[index_lb]) / (mid[index_lb] - worst[index_lb]))
                if Ya < 1.5: Ya = 0.5
                modified = Ya * mid + ((1-Ya) * worst)
        
            if np.any(index_ub): # upper bound violation
                Ya = 1 + ((self.ub[index_ub] - mid[index_ub]) / (mid[index_ub] - worst[index_ub]))
                if Ya < 1.5: Ya = 0.5
                modified = Ya * mid + ((1-Ya) * worst)
            
            index_lb = suggestion < self.lb
            index_ub = suggestion > self.ub
        
        return modified, Ya

    def __Yopt_calc(self, w, b, p, r):
        if self.Ya == None: # R was not constrained previously
            if (w-(2*p)+r) < 0:
                Yopt = ((w - p)/(w - (2 * p) + r)) + 0.5
            elif r < b: # reflection better than best, expansion
                Yopt = 3
            elif r < w: # reflection better than worst, but worse that best, outside contraction
                Yopt = 1.5
            else: # inside contraction
                Yopt = 0.5
        else: # R was constrained
            if self.Ya > 1: # Boundary in +ve direction
                if p > ((r + ((self.Ya - 1) * w)) / self.Ya):
                    Yopt = (0.5 * (self.Ya**2 - self.Ya) * (w - p) / ((self.Ya - 1) * w - (self.Ya * p) + r)) + 0.5
                else:
                    Yopt = self.YoptMin
            else: # Boundary in -ve direction
                if p < ((r + ((self.Ya - 1) * w)) / self.Ya):
                    Yopt = (0.5 * (self.Ya**2 - self.Ya) * (w - p) / ((self.Ya - 1) * w - (self.Ya * p) + r)) + 0.5
                else:
                    Yopt = self.YoptMin
        
        # Check Yopt is acceptable
        if np.abs(Yopt - 1) < self.YoptMargin: # Optimisation can stall if Yopt near 1 or 0
            if Yopt > 1: Yopt = 1 + self.YoptMargin
            else: Yopt = 1 - self.YoptMargin
        elif np.abs(Yopt) < self.YoptMargin:
            if Yopt > 0: Yopt = 0 + self.YoptMargin
            else: Yopt = 0 - self.YoptMargin
        else:
            Yopt = min(Yopt, self.YoptMax)
            Yopt = max(Yopt, self.YoptMin)

        return Yopt

    def __cov_calc(self, responses):
        """Calculate COV for termination criteria"""
        return 100 * np.std(responses) / np.mean(responses)


    def suggest(self, y=None):
        """
        Suggest conditions if user is running a batch optimisation
         Initial -> midpoint (don't request, just calc) -> reflection (request) -> optimum(request) -> calc next initial and loop back to midpoint (also check if finished)
        """
        if self.X == None:
            ncurrent = 0
        else:
            ncurrent = np.shape(self.X[-1])[0]

        if ncurrent <= 0 and y == None: # Check if very first call
            self.Xnext = self.__initial()
            self.X = [self.Xnext]
            return self.Xnext
        elif ncurrent == (self.nvar + 3): # Calculate retention and move onto next 'simplex' iteration
            self.y[-1] = np.hstack((self.y[-1], y))
            ind = np.argsort(self.y[-1])
            y = self.y[-1][ind[:3]]
            
            if self.__cov_calc(y) < self.covLimit:
                return None
            self.X.append(self.X[-1][ind[:3]])
            ncurrent = (self.nvar + 1)

        if ncurrent == (self.nvar + 1): # Midpoint calculation and reflection request
            self.y.append(y)
            # Sort outputs to determine midpoint
            self.ind = np.argsort(self.y[-1])

            self.mid = np.mean(self.X[-1][self.ind[:-1]],axis=0)
            self.mid_p = np.mean(self.y[-1][self.ind[:-1]])

            self.worst = self.X[-1][self.ind[-1]]

            reflect = 2 * self.mid - self.worst

            self.Xnext, self.Ya = self.__boundary_check(reflect, self.mid, self.worst)
            self.X[-1] = np.vstack((self.X[-1], self.Xnext))

            return self.Xnext
            
        elif ncurrent == (self.nvar + 2): # Optimum calculation and request
            w = self.y[-1][self.ind[-1]]
            b = self.y[-1][self.ind[0]]
            p = self.mid_p
            r = y

            Yopt = self.__Yopt_calc(w, b, p, r)

            optimum = Yopt * self.mid + ((1 - Yopt) * self.worst)

            self.Xnext, self.Ya = self.__boundary_check(optimum, self.mid, self.worst)

            self.X[-1] = np.vstack((self.X[-1], self.Xnext))
            self.y[-1] = np.hstack((self.y[-1], y))

            return self.Xnext

        raise NotImplementedError

    def optimise(self, func):
        """Closed loop optimisation of a given function, will only output optimisation result to user"""
        raise NotImplementedError       


class Tests(unittest.TestCase):
    def __init__(self):
        self.initialexpected = np.array([[1, 2], [1.05, 2], [1, 2.1]])
        smsim = SMSIM([[0,0],[3,3]], x0=[1,2])
        self.initialresult = smsim.suggest()

    def test_initial(self):
        assert(self.initialexpected == self.initialresult).all(), "Initial condition failure"
    

if __name__ == "__main__":
    a = 5
    test = Tests()
    test.test_initial()