import numpy as np

def matyas(X):
    return (0.26 * (X[:,0] ** 2 + X[:,1] ** 2) - 0.48 * X[:,0] * X[:,1]).reshape(-1)

if __name__ == "__main__":
    print(matyas(np.array([1,1]).reshape(1,-1)))
    print(matyas(np.array([0,0]).reshape(1,-1)))
    print(matyas(np.array([[0,0],[1,1]]).reshape(-1,2)))
    print(matyas(np.array([0,0]).reshape(1,-1)).shape)
    print(matyas(np.array([[0,0],[1,1]]).reshape(-1,2)).shape)