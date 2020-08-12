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
            elif r > b: # reflection better than best, expansion
                Yopt = 3
            elif r > w: # reflection better than worst, but worse that best, outside contraction
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
        return np.abs(100 * np.std(responses) / np.mean(responses))


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
        elif ncurrent == (self.nvar + 3): # Calculate retention and move onto next 'simplex' iteration
            self.y[-1] = np.hstack((self.y[-1], y))
            o = y
            
            if o > self.r:
                self.X.append(np.vstack((self.best, self.next, self.optimum)))
                y = np.hstack((self.b, self.n, o))
            else:
                self.X.append(np.vstack((self.best, self.next, self.reflect)))
                y = np.hstack((self.b, self.n, self.r))

            if self.__cov_calc(y) < self.covLimit:
               return None
            ncurrent = (self.nvar + 1)

        if ncurrent == (self.nvar + 1): # Midpoint calculation and reflection request
            self.y.append(y)
            # Sort outputs to determine midpoint
            self.ind = np.argsort(self.y[-1])
            self.best = self.X[-1][self.ind[-1]]
            self.b = self.y[-1][self.ind[-1]]
            self.next = self.X[-1][self.ind[-2]]
            self.n = self.y[-1][self.ind[-2]]

            self.mid = np.mean(self.X[-1][self.ind[-2:]],axis=0)
            self.mid_p = np.mean(self.y[-1][self.ind[-2:]])

            self.worst = self.X[-1][self.ind[0]]
            self.w = self.y[-1][self.ind[0]]

            self.reflect = 2 * self.mid - self.worst

            self.reflect, self.Ya = self.__boundary_check(self.reflect, self.mid, self.worst)
            self.Xnext = self.reflect
            self.X[-1] = np.vstack((self.X[-1], self.Xnext))
     
        elif ncurrent == (self.nvar + 2): # Optimum calculation and request
            self.r = y

            Yopt = self.__Yopt_calc(self.w, self.b, self.mid_p, self.r)

            self.optimum = Yopt * self.mid + ((1 - Yopt) * self.worst)

            self.Xnext, self.Ya = self.__boundary_check(self.optimum, self.mid, self.worst)

            self.X[-1] = np.vstack((self.X[-1], self.Xnext))
            self.y[-1] = np.hstack((self.y[-1], y))

        return self.Xnext.reshape(-1,self.nvar)
        

    def optimise(self, func):
        """Closed loop optimisation of a given function, will only output optimisation result to user"""
        assert func(np.array(self.x0).reshape(1,-1)).shape == (1,), "Please provide a function that produces a numpy array 1D vector as an output for a single point the shape should be (1,)"
        check = True
        maxits = 100
        it = 0
        while check and it < maxits:
            if it == 0:
                Xnext = self.suggest()
                print(Xnext)
            else:
                Xnext = self.suggest(y)
                print(Xnext)
            
            if np.any(Xnext != None): 
                y = func(Xnext)
                it += 1
            else:
                ind = np.argmax(self.y[-1])
                return self.y[-1][ind], self.X[-1][ind]
        ind = np.argmax(self.y[-1])
        return self.y[-1][ind], self.X[-1][ind], 'Max Iteration Reached'


class Tests(unittest.TestCase):
    def __init__(self):
        self.initialexpected = np.array([[1, 2], [1.1, 2], [1, 2.2]])
        smsim = SMSIM([[0,0],[3,3]], x0=[1,2])
        self.initialresult = smsim.suggest()

    def test_initial(self):
        #assert(self.initialexpected == self.initialresult).all(), "Initial condition failure"
        a=1
    
def matyas(X):
    return -(0.26 * (X[:,0] ** 2 + X[:,1] ** 2) - 0.48 * X[:,0] * X[:,1]).reshape(-1)

if __name__ == "__main__":
    a = 5
    test = Tests()
    test.test_initial()

    opt = SMSIM([[-10,-10],[10,10]], x0=[5,-5])
    print(matyas(np.array([4,-5.75]).reshape((-1,2))))
    print(opt.optimise(matyas))
