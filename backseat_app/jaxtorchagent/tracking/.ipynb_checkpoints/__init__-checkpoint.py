from .least_squares import LSTracker
from .particle_filter import ParticleFilter
import math

class Tracker:
    """wrapper for choosing between particle filter and least_squares tracking"""
    
    def __init__(self, method='ls', dt=60, **kwargs):
        
        assert method in ['ls', 'pf'], "available methods are 'ls' (Least Squares) and 'pf' (Particle Filter)"
        self.method = method
        self.dt = dt

        # default prediction
        self.pred = [0, 0, 0]
        
        if self.method == 'ls':
            self.model = LSTracker(**kwargs)
        else:
            self.model = ParticleFilter(**kwargs)
            
    def update_and_predict(self, ranges, positions, dt=None):
        
        # default time difference
        if dt is None:
            dt = self.dt

        # Least Squares method
        if self.method == 'ls':
            for r, pos in zip(ranges, positions):
                self.model.add_range(z=r, pos=pos)
            pred_xy = self.model.predict(positions[0])
        
        # Particle Filter method
        else:
            for r, pos in zip(ranges, positions):
                pred_xy = self.model.update_and_predict(dt=dt, z=r, pos=pos)

        # update pred
        self.pred[:2] = pred_xy
        z = self.estimate_z(r, pos, pred_xy)
        if z is not None:
            self.pred[2] = z

        return self.pred
    
    def estimate_z(self, r, pos, pred):
        """for now z is estimated using basic geometry"""
        to_square = r**2 - (pos[0] - pred[0])**2 - (pos[1] - pred[1])**2
        if to_square < 0:
            return None
        else:
            return pos[2] + math.sqrt(to_square)