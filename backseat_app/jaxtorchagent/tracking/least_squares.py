from collections import deque
import numpy as np

class LSTracker(object):
    
    def __init__(self, num_points=30):
        self.lsxs = None
        self.Plsu = None 
        self.eastingpoints_LS = deque(maxlen=num_points)
        self.northingpoints_LS = deque(maxlen=num_points)
        self.allz = deque(maxlen=num_points)
        self.N = np.identity(3)[:-1]

    def reset(self):
        self.lsxs = None
        self.Plsu = None 
        self.eastingpoints_LS.clear()
        self.northingpoints_LS.clear()
        self.allz.clear()
        self.N = np.identity(3)[:-1]
        
    def add_range(self,z,pos,update=False):
        self.allz.append(z)
        self.eastingpoints_LS.append(pos[0])
        self.northingpoints_LS.append(pos[1])
        if update:
            self.update()

    def get_pred(self):
        if self.lsxs is None:
            raise ("You need to update the model at least once before getting the predictions")
        else:
            return self.lsxs
        
    def predict(self, pos):
        numpoints = len(self.allz)
        if numpoints > 3:
            # N can be precomputed
            N = self.N
            # P
            P = np.array([self.eastingpoints_LS,self.northingpoints_LS])
            # A
            A = np.full((numpoints,3),-1, dtype=float)
            A[:,0] = P[0]*2
            A[:,1] = P[1]*2
            #b 
            allz_np = np.array(self.allz)
            b = (np.diag(np.matmul(P.T,P))-(allz_np*allz_np)).reshape(numpoints, 1)

            try:
                # N*(A.T*A).I*A.T*b
                self.Plsu = np.matmul(np.matmul(np.matmul(N,np.linalg.inv(np.matmul(A.T,A))), A.T), b)
            except:
                print('WARNING: LS singular matrix')
                try:
                    self.Plsu = np.matmul(np.matmul(np.matmul(N,np.linalg.inv(np.matmul(A.T,A+1e-6))), A.T), b)
                except:
                    pass
                
        if self.lsxs is None:
            self.lsxs = np.array([pos[0], pos[1]])

        if self.Plsu is not None:
            self.lsxs  = np.array([self.Plsu.item(0),self.Plsu.item(1)])
        
        return self.get_pred()