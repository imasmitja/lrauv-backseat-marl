import math
import numpy as np

from .least_squares import LSTracker
from .particle_filter import ParticleFilter




class Tracker_ivan:
    """wrapper for choosing between particle filter and least_squares tracking"""
    
    def __init__(self, method='ls', dt=30, pf_seed=0, **kwargs):
        
        assert method in ['ls', 'pf'], "available methods are 'ls' (Least Squares) and 'pf' (Particle Filter)"
        self.method = method
        self.dt = dt

        # default prediction
        self.pred = [0, 0, 0]
        
        if self.method == 'ls':
            self.model = LSTracker(**kwargs)
        else:
            self.model = ParticleFilter(**kwargs)
            self.pf_state = None

    def reset(self):
        if self.method == 'ls':
            self.model.reset()
        else:
            self.pf_state = None
            
    def update_and_predict(self, ranges, positions, depth=None, dt=None):
        
        # default time difference
        if dt is None:
            dt = self.dt

        # bring the ranges to 2d if the landmarks depth is known
        if depth is not None:
            for i, (r, pos) in enumerate(zip(ranges, positions)):
                if r != 0 and ((pos[2] - depth)**2)<=r**2 :
                    ranges[i] = np.sqrt(r**2 - (pos[2] - depth)**2)

        # Least Squares method
        if self.method == 'ls':
            for r, pos in zip(ranges, positions):
                self.model.add_range(z=r, pos=pos)
            pred_xy = self.model.predict(positions[0])
        
        # Particle Filter method
        else:
            for r, pos in zip(ranges,positions):
                if self.pf_state is None:
                    # initialize the particle filter if it's the first time
                    self.model.init_particles(position=pos, slantrange=r)
                    self.pf_state = True
                pos = np.array(pos)[:2] # only x, y
                r = np.array(r)
                pred_xy = self.model.update_and_predict(
                                            dt=dt, z=r, pos=pos)

        # if the prediction is not available, use the first position
        if np.isnan(pred_xy).any():
            pred_xy = [positions[0][0], positions[0][1]]
        
        # convert to float list
        pred_xy = [float(x) for x in pred_xy]

        # update pred
        self.pred[:2] = pred_xy

        # estimate z if the depth is not known
        if depth is None:
            z = self.estimate_z(ranges[0], positions[0], pred_xy)
            if z is not None:
                self.pred[2] = z
        else:
            self.pred[2] = depth

        return self.pred
    
    def estimate_z(self, r, pos, pred):
        """for now z is estimated using basic geometry"""
        to_square = r**2 - (pos[0] - pred[0])**2 - (pos[1] - pred[1])**2
        if to_square < 0:
            return None
        else:
            return pos[2] + math.sqrt(to_square)