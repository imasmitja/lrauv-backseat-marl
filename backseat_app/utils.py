# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:50:18 2022

@author: Ivan Masmitja Rusinol

Project: AIforUTracking
"""

import numpy as np
import random
import time
import utm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from backseat_app.jaxtorchagent.production_agent import load





SOUND_SPEED = 1500.

#############################################################
## Particle Filter
############################################################
#For modeling the target we will use the TargetClass with the following attributes 
#and functions:
class ParticleFilter(object):
    """ Class for the Particle Filter """
 
    def __init__(self,std_range,init_velocity,dimx,particle_number = 6000, method = 'range', max_pf_range = 250):
 
        self.std_range = std_range
        self.init_velocity = init_velocity 
        self.x = np.zeros([particle_number,dimx])
        self.oldx = np.zeros([particle_number,dimx])
        self.particle_number = particle_number
        
        self._x = np.zeros([dimx])
       
        # target's noise
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.velocity_noise = 0.0
        
        # time interval
        self.dimx=dimx
        
        self._velocity = 0
        self._orientation = 0
        
        #Weights
        self.w = np.ones(particle_number)
        
        #Covariance of the result
        self.covariance_vals = [100,100]
        self.covariance_theta = 0.
        
        #Flag to initialize the particles
        self.initialized = False
        
        #save actual data as a old to be used on TDOA method
        self.measurement_old = 0
        self.dist_all_old = np.zeros(particle_number)
        self.w_old = self.w
        self.observer_old = np.array([0,0,0,0])
        
        self.method = method
        #covariance matrix of final estimation
        self.cov_matrix = np.ones([2,2])
        
        #maximum target range
        self.max_pf_range = max_pf_range
        
        self.previous_observer = []
        self.previous_z = 0
        
        
    def target_estimation(self):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        #8- Target prediction (we predict the best estimation for target's position = mean of all particles)
        sumx = 0.0
        sumy = 0.0
        sumvx = 0.0
        sumvy = 0.0

        method = 2
        if method == 1:
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        if method == 2:
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]
            if np.sum(self.w) == 0. or np.isnan(np.sum(self.w)):
                self._x = np.array([sumx, sumvx, sumy, sumvy])/1e-12
            else:
                self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            
            # #new approach to find the colosest particle to the mean
            # x_pos = np.where(abs(self.x.T[0]-self._x[0]) == np.amin(abs(self.x.T[0]-self._x[0])))[0][0]
            # y_pos = np.where(abs(self.x.T[2]-self._x[2]) == np.amin(abs(self.x.T[2]-self._x[2])))[0][0]
            # x_mean = (self.x.T[0][x_pos] + self.x.T[0][y_pos])/2.
            # y_mean = (self.x.T[2][x_pos] + self.x.T[2][y_pos])/2.
            # self._x[0] = x_mean
            # self._x[2] = y_mean
            
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        #finally the covariance matrix is computed. 
        #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        xarray = self.x.T[0]
        yarray = self.x.T[2]
        self.cov_matrix = np.cov(xarray, yarray)
        return

    def init_particles(self,position,slantrange):
    	
        print('position=',position, ' slantrange=',slantrange)
        for i in range(self.particle_number):
            #Random distribution with circle shape
            t = 2*np.pi*np.random.rand()
            if self.method == 'area':
                r = np.random.rand()*self.max_pf_range*2 - self.max_pf_range
            else:
                r = np.random.rand()*self.std_range*2 - self.std_range + slantrange
            
            self.x[i][0] = r*np.cos(t)+position[0]
            self.x[i][2] = r*np.sin(t)+position[2]
            #target's orientation
            orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
            # target's velocity 
            v = random.gauss(self.init_velocity, self.init_velocity/2)  
            self.x[i][1] = np.cos(orientation)*v
            self.x[i][3] = np.sin(orientation)*v
            #init particle weights
            self.w[i] = 1./self.particle_number
        self.target_estimation()
        self.initialized = True
        print('WARNING: Particles initialized')
        return
    
    #Noise parameters can be set by:
    def set_noise(self, forward_noise, turn_noise, sense_noise, velocity_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """
        # target's noise
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.velocity_noise = velocity_noise

    #Move particles acording to its motion
    def predict(self,dt):
        """ Perform target's turn and move
        :param turn:    turn command
        :param forward: forward command
        :return target's state after the move
        """
        gaussnoise = False
        for i in range(self.particle_number):
            # turn, and add randomness to the turning command
            turn = np.arctan2(self.x[i][3],self.x[i][1])
            if gaussnoise == True:
                orientation = turn + random.gauss(0.0, self.turn_noise)
            else:
                orientation = turn +  np.random.rand()*self.turn_noise*2 -self.turn_noise
            orientation %= 2 * np.pi
         
            # move, and add randomness to the motion command
            velocity = np.sqrt(self.x[i][1]**2+self.x[i][3]**2)
            forward = velocity*dt
            if gaussnoise == True:
                dist = float(forward) + random.gauss(0.0, self.forward_noise)
            else:
                dist = float(forward) + np.random.rand()*self.forward_noise*2 - self.forward_noise
            self.x[i][0] = self.x[i][0] + (np.cos(orientation) * dist)
            self.x[i][2] = self.x[i][2] + (np.sin(orientation) * dist)
            if gaussnoise == True:
                newvelocity = velocity + random.gauss(0.0, self.velocity_noise)
            else:
                newvelocity = velocity + np.random.rand()*self.velocity_noise*2 - self.velocity_noise
            if newvelocity < 0:
                newvelocity = 0
            self.x[i][1] = np.cos(orientation) * newvelocity
            self.x[i][3] = np.sin(orientation) * newvelocity
        return 

    #To calculate Gaussian probability:
    @staticmethod
    def gaussian(self,mu_old,mu, sigma, z_old,z,inc_observer):
        """ calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        :param mu:    distance to the landmark
        :param sigma: standard deviation
        :param x:     distance to the landmark measured by the target
        :return gaussian value
        """
        if self.method == 'area':
            sigma = 1. #was 5
            particlesRange = self.max_pf_range 
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma in a filled circle shape
            # We use the Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution)
            if z != -1: #a new ping is received -> #all particles outside the tagrange have a small weight; #all particles inside the tagrange have a big weight
                return (1/2.)-(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
            else: #no new ping is received -> #all particles outside the tagrange have a big weight; #all particles inside the tagrange have a small weight
                sigma = 40.
                return (1/2.)+(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
        else:
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
            return np.exp(- ((mu - z) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))
    
    #The next function we will need to assign a weight to each particle according to 
    #the current measurement. See the text below for more details. It uses effectively a 
    #Gaussian that measures how far away the predicted measurements would be from the 
    #actual measurements. Note that for this function you should take care of measurement 
    #noise to prevent division by zero. Such checks are skipped here to keep the code 
    #as short and compact as possible.
    def measurement_prob(self, measurement,observer,error_mult = 1):
        """ Calculate the measurement probability: how likely a measurement should be
        :param measurement: current measurement
        :return probability
        """
        #The closer a particle to a correct position, the more likely will be the set of 
            #measurements given this position. The mismatch of the actual measurement and the 
            #predicted measurement leads to a so-called importance weight. It tells us how important 
            #that specific particle is. The larger the weight, the more important it is. According 
            #to this each of our particles in the list will have a different weight depending on 
            #a specific target measurement. Some will look very plausible, others might look 
            #very implausible.           
        dist_all = []
        equal = 0
        for i in range(self.particle_number):
            dist = np.sqrt((self.x[i][0] - observer[0])**2 + (self.x[i][2] - observer[2])**2)
            dist_old = np.sqrt((self.x[i][0] - self.observer_old[0])**2 + (self.x[i][2] - self.observer_old[2])**2)
            inc_observer = np.sqrt((observer[0] - self.observer_old[0])**2 + (observer[2] - self.observer_old[2])**2)
            self.w[i] = self.gaussian(self,dist_old,dist, self.sense_noise*error_mult, self.measurement_old,measurement,inc_observer)
            inc_mu = (self.dist_all_old[i]-dist)
            inc_z = (self.measurement_old-measurement)
            if (inc_mu >= 0 and inc_z >= 0) or (inc_mu < 0 and inc_z < 0):
                equal +=1
            dist_all.append(dist)
            
        #save actual data as a old to be used on TDOA method
        self.measurement_old = measurement
        self.dist_all_old = np.array(dist_all)
        self.w_old=self.w
        self.observer_old = observer
        return 
    
    def resampling(self,z,method = 2):
        #After that we let these particles survive randomly, but the probability of survival 
            #will be proportional to the weights.
            #The final step of the particle filter algorithm consists in sampling particles from 
            #the list p with a probability which is proportional to its corresponding w value. 
            #Particles in p having a large weight in w should be drawn more frequently than the 
            #ones with a small value
            #Here is a pseudo-code of the resampling step:
            #while w[index] < beta:
            #    beta = beta - w[index]
            #    index = index + 1
            #    select p[index]
                        
        #method = 2 #NO compound method
        #method = 3.2 #compound method
        
        if self._x[0] == 0 and self._x[2] == 0:
            method = 2
        else:
            method = 3 #compound method presented in OCEANS'18 Kobe
        
        if method == 1:   
            # 4- resampling with a sample probability proportional
            # to the importance weight
            p3 = np.zeros([self.particle_number,self.dimx])
            index = int(np.random.random() * self.particle_number)
            beta = 0.0
            mw = max(self.w)
            for i in range(self.particle_number):
                beta += np.random.random() * 2.0 * mw
                while beta > self.w[index]:
                    beta -= self.w[index]
                    index = (index + 1) % self.particle_number
                p3[i]=self.x[index]
            self.x = p3
            return
        if method == 2:
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            # Systematic Resampling
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            if np.sum(self.w) == 0. or np.isnan(np.sum(self.w)):
                normalized_w = self.w/1e-12
            else:
                normalized_w = self.w/np.sum(self.w)
            
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/self.particle_number
            i = 0
            for j in range(self.particle_number):
                while (u > ci[i]):
                    i += 1
                    if i==self.particle_number:
                        i-=1
                        break
                p3[j]=self.x[i]
                u = u + 1./self.particle_number
            self.x = p3
            return
        if method == 3: #this mehtod works ok and was presented in OCEANS Kobe 2018
            # Systematic Resampling + random resampling
            if self.particle_number == 10000:
                ratio = 340 #160 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 6000:
                ratio = 400 #100 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 3000:
                ratio = 200 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 1000:
                ratio = 120 #15 works ok; ratio=10 is ok for statik targets
            else:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            radii = 50. #50 works ok
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            # if np.sum(self.w) > 1e-12:
            #     normalized_w = self.w/np.sum(self.w)
            # else:
            #     normalized_w = self.w/1e-6
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i+1]= aux
                self.w[j+i+1] = 1./(self.particle_number/3.)
            self.x = p3
            return
        if method == 3.2: 
            #this mehtod is a modification used in TAG-Only tracking, is similar than the method presented in OCEANS Kobe 2018
            #the main difference is that the random resampling is centred over the WG position instead of the Target estimation
            # Systematic Resampling + random resampling
            ratio = 50 #50 works ok
            radii = self.max_pf_range #50 works ok
            
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                i += 1
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self.observer_old[0]
                aux[2] = r*np.sin(t)+self.observer_old[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i]= aux
                self.w[j+i] = 1/10000.
            self.x = p3
            return
    
    
    #6- It computes the average error of each particle relative to the target pose. We call 
            #this function at the end of each iteration:
            # here we get a set of co-located particles   
    #At every iteration we want to see the overall quality of the solution, for this 
    #we will use the following function:
    def evaluation(self,observer,z,max_error=50):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        if self.method != 'area':
        
            #Evaluate the distance error
            sum2 = 0.0
            for i in range(self.particle_number):
                # Calculate the mean error of the system between Landmark (WG) and particle set
                dx = (self.x[i][0] - observer[0])
                dy = (self.x[i][2] - observer[2])
                err = np.sqrt(dx**2 + dy**2)
                sum2 += err
            # print('Evaluation -> distance error: ',abs(sum2/self.particle_number - z))
            
            #Evaluate the covariance matrix
            err_x = self.x.T[0]-self._x[0]
            err_y = self.x.T[2]-self._x[2]
            cov = np.cov(err_x,err_y)
            # Compute eigenvalues and associated eigenvectors
            vals, vecs = np.linalg.eig(cov)
            confidence_int = 2.326**2
            if vals[0]<0 or vals[1]<0 or np.isnan(vals[0]) == True or np.isnan(vals[1]) == True:
                vals = [0.,0.]
            self.covariance_vals = np.sqrt(vals) * confidence_int
            # Compute tilt of ellipse using first eigenvector
            vec_x, vec_y = vecs[:,0]
            self.covariance_theta = np.arctan2(vec_y,vec_x)
            # print('Evaluation -> covariance (CI of 98): %.2f m(x) %.2f m(y) %.2f deg'%(self.covariance_vals[0],self.covariance_vals[1],np.degrees(self.covariance_theta)))
            print('Evaluation -> covariance (CI of 98): %.2f '%(np.sqrt(self.covariance_vals[0]**2+self.covariance_vals[1]**2)))
            print('errorPF=',abs(sum2/self.particle_number - z))
            print('Evaluaiton -> max(self.w)=%.4f sum(self.w)=%.4f'%(np.max(self.w),np.sum(self.w)))
            if abs(sum2/self.particle_number - z) > max_error:
                self.initialized = False
                self.init_particles(position=observer, slantrange=z)
                # self.init_particles(position=self.previous_observer, slantrange=self.previous_z)
                # self.measurement_prob(measurement=z,observer=observer,error_mult=50.) 
        else:
            if np.max(self.w) < 0.1:
                self.initialized = False
            #Compute maximum particle dispersion:
            max_dispersion = np.sqrt((np.max(self.x.T[0])-np.min(self.x.T[0]))**2+(np.max(self.x.T[2])-np.min(self.x.T[2]))**2)     
        return 


##########################################################################################################
##############################                    TARGET CLASS   ##########################################
###########################################################################################################
class Target(object):
    def __init__(self,method='range',max_pf_range=250):
        #Target parameters
        self.method = method
        
        ############## PF initialization #######################################################################
        #Our particle filter will maintain a set of n random guesses (particles) where 
        #the target might be. Each guess (or particle) is a vector containing [x,vx,y,vy]
        # create a set of particles
        # sense_noise is not used in area-only
        # self.pf = ParticleFilter(std_range=.01,init_velocity=.001,dimx=4,particle_number=6000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.0001, turn_noise = 0.1, sense_noise=.05, velocity_noise = 0.0001)
        
        # self.pf = ParticleFilter(std_range=.005,init_velocity=.001,dimx=4,particle_number=1000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=.09, velocity_noise = 0.0001)
        
        # self.pf = ParticleFilter(std_range=.02,init_velocity=.2,dimx=4,particle_number=1000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=.005, velocity_noise = 0.01)
        
        # self.pf = ParticleFilter(std_range=.02,init_velocity=.2,dimx=4,particle_number=1000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=.005, velocity_noise = 0.01)
        
        #as BSC RL tests in Python
        self.pf = ParticleFilter(std_range=20.,init_velocity=0.4,dimx=4,particle_number=10000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = .5, sense_noise=2., velocity_noise = 0.01)
        self.pf.set_noise(forward_noise = 0.01, turn_noise = .9, sense_noise=5., velocity_noise = 0.01)
            
        self.position = 0
        
        #############LS initialization###########################################################################
        self.lsxs=[]
        self.eastingpoints_LS=[]
        self.northingpoints_LS=[]
        self.Plsu=np.array([])
        self.allz=[]
        #self.ls_position = 0
    
    #############################################################################################
    ####            Particle Filter Algorithm  (PF)                                             ##         
    #############################################################################################                               
    def updatePF(self,dt,new_range,z,myobserver,update=True):
        max_error = 150.
        if update == True:
            print ('myobserver=',myobserver, '. Range=',z )
            # Initialize the particles if needed
            if self.pf.initialized == False:
                self.pf.init_particles(position=myobserver, slantrange=z)
                
            #we save the current particle positions to plot as the old ones
            self.pf.oldx = self.pf.x.copy() 
            
            # Predict step (move all particles)
            self.pf.predict(dt)
            
            # Update step (weight and resample)
            if new_range == True:     
                # Update the weiths according its probability
                self.pf.measurement_prob(measurement=z,observer=myobserver)   
                # if weights are equal to 0 means that the estimation is not going well and we reset the values using the previous measurement
                if max(self.pf.w) < 0.00099 or np.sqrt(self.pf.covariance_vals[0]**2+self.pf.covariance_vals[1]**2) < 50.:
                    print('WARNIN, particle w too low, initializing again')
                    if self.pf.previous_observer == []: #initialize with the first measurement
                        self.pf.previous_observer = myobserver.copy()
                        self.pf.previous_z = z +0.
                    self.pf.init_particles(position=self.pf.previous_observer, slantrange=self.pf.previous_z)
                    self.pf.measurement_prob(measurement=z,observer=myobserver,error_mult=50.) 
                self.pf.previous_observer = myobserver.copy()
                self.pf.previous_z = z +0.
                #Resampling        
                self.pf.resampling(z)
                # Calculate the avarage error. If it's too big the particle filter is initialized                    
                self.pf.evaluation(observer=myobserver,z=z,max_error=max_error)    
            # We compute the average of all particles to fint the target
            self.pf.target_estimation()
        
        if self.pf.initialized == False:
            self.position = np.array([myobserver.item(0),0.,myobserver.item(2),0.])
        else:
            #Save position
            self.position = self.pf._x.copy()
        return True

    #############################################################################################
    ####             Least Squares Algorithm  (LS)                                             ##         
    #############################################################################################
    def updateLS(self,dt,new_range,z,myobserver):
        num_ls_points_used = 10
        #Propagate current target state estimate
        if new_range == True:
            self.allz.append(z)
            self.eastingpoints_LS.append(myobserver[0])
            self.northingpoints_LS.append(myobserver[2])
        numpoints = len(self.eastingpoints_LS)
        if numpoints > 3:
            #Unconstrained Least Squares (LS-U) algorithm 2D
            #/P_LS-U = N0* = N(A^T A)^-1 A^T b
            #where:
            P=np.matrix([self.eastingpoints_LS[-num_ls_points_used:],self.northingpoints_LS[-num_ls_points_used:]])
            # N is:
            N = np.concatenate((np.identity(2),np.matrix([np.zeros(2)]).T),axis=1)
            # A is:
            num = len(self.eastingpoints_LS[-num_ls_points_used:])
            A = np.concatenate((2*P.T,np.matrix([np.zeros(num)]).T-1),axis=1)
            # b is:
            b = np.matrix([np.diag(P.T*P)-np.array(self.allz[-num_ls_points_used:])*np.array(self.allz[-num_ls_points_used:])]).T
            # Then using the formula "/P_LS-U" the position of the target is:
            try:
                self.Plsu = N*(A.T*A).I*A.T*b
            except:
                print('WARNING: LS singular matrix')
                try:
                    self.Plsu = N*(A.T*A+1e-6).I*A.T*b
                except:
                    pass
            # Finally we calculate the depth as follows
#                r=np.matrix(np.power(allz,2)).T
#                a=np.matrix(np.power(Plsu[0]-eastingpoints_LS,2)).T
#                b=np.matrix(np.power(Plsu[1]-northingpoints_LS,2)).T
#                depth = np.sqrt(np.abs(r-a-b))
#                depth = np.mean(depth)
#                Plsu = np.concatenate((Plsu.T,np.matrix(depth)),axis=1).T
            #add offset
#                Plsu[0] = Plsu[0] + t_position.item(0)
#                Plsu[1] = Plsu[1] + t_position.item(1)
#                eastingpoints = eastingpoints + t_position.item(0)
#                northingpoints = northingpoints + t_position.item(1)
            #Error in 'm'
#                error = np.concatenate((t_position.T,np.matrix(simdepth)),axis=1).T - Plsu
#                allerror = np.append(allerror,error,axis=1)

        #Compute MAP orientation and save position
        try:
            ls_orientation = np.arctan2(self.Plsu[1]-self.lsxs[-1][2],self.Plsu[1]-self.lsxs[-1][0])
        except IndexError:
            ls_orientation = 0
        try:
            ls_velocity = np.array([(self.Plsu[0]-self.lsxs[-1][0])/dt,(self.Plsu[1]-self.lsxs[-1][1])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        try:
            ls_position = np.array([self.Plsu.item(0),ls_velocity.item(0),self.Plsu.item(1),ls_velocity.item(1),ls_orientation.item(0)])
        except IndexError:
            ls_position = np.array([myobserver[0],ls_velocity[0],myobserver[2],ls_velocity[1],ls_orientation])
        self.lsxs.append(ls_position)
        #Save position
        self.position = ls_position
        return True

###########################################################################################################
##############################      Reinforcement Learning Actor                 ##########################
###########################################################################################################  
#writting a numpy imnplementation of a Torch actor.
class np_rl_agent(object):
    def __init__(self):
        #w1(10x64)
        #w2(64x32)
        #w3(31x1)
        self.w1 = np.loadtxt('./backseat_app/pretrined_agents/w1.txt',delimiter=',').T
        self.w2 = np.loadtxt('./backseat_app/pretrined_agents/w2.txt',delimiter=',').T
        self.w3 = np.loadtxt('./backseat_app/pretrined_agents/w3.txt',delimiter=',').T
        #bias
        self.b1 = np.loadtxt('./backseat_app/pretrined_agents/b1.txt',delimiter=',').T
        self.b2 = np.loadtxt('./backseat_app/pretrined_agents/b2.txt',delimiter=',').T
        self.b3 = np.loadtxt('./backseat_app/pretrined_agents/b3.txt',delimiter=',').T
        
#        print('w1',self.w1)
#        print('w1',self.w2)
#        print('w1',self.w3)
#        print('w1',self.b1)
#        print('w1',self.b2)
#        print('w1',self.b3)
        
        
    #activation function
    def np_relu(self,x):
        for i in range(x.size):
            if x[i]<0.:
                x[i] = 0.
        return x
    
    def np_forward(self,x):
        #layer1 (input)
        prob1 = x.dot(self.w1) + self.b1
        prob1 = self.np_relu(prob1)
        
        #layer2 (hiden)
        prob2 = prob1.dot(self.w2) + self.b2
        prob2 = self.np_relu(prob2)
        
        #layer3 (output)
        mean = prob2.dot(self.w3) + self.b3
        
        return mean.clip(-1.0,1.0)

class np_rnn_rl_agent(object):
    def __init__(self):
        #values for agent: sacqmix_l_v10test_lstm_emofish
        #weights
        self.w1rnn = np.loadtxt('./backseat_app/pretrined_agents/w1rnn.txt',delimiter=' ').T
        self.w2rnn = np.loadtxt('./backseat_app/pretrined_agents/w2rnn.txt',delimiter=' ').T
        self.w3rnn = np.loadtxt('./backseat_app/pretrined_agents/w3rnn.txt',delimiter=' ').T
        #bias
        self.b1rnn = np.loadtxt('./backseat_app/pretrined_agents/b1rnn.txt',delimiter=' ').T
        self.b2rnn = np.loadtxt('./backseat_app/pretrined_agents/b2rnn.txt',delimiter=' ').T
        self.b3rnn = np.loadtxt('./backseat_app/pretrined_agents/b3rnn.txt',delimiter=' ').T
        
        #LSTM parameters
        #ft
        self.whf = np.loadtxt('./backseat_app/pretrined_agents/whf.txt',delimiter=' ').T
        self.wif = np.loadtxt('./backseat_app/pretrined_agents/wif.txt',delimiter=' ').T
        self.bhf = np.loadtxt('./backseat_app/pretrined_agents/bhf.txt',delimiter=' ').T
        self.bif = np.loadtxt('./backseat_app/pretrined_agents/bif.txt',delimiter=' ').T
        #it
        self.whi = np.loadtxt('./backseat_app/pretrined_agents/whi.txt',delimiter=' ').T
        self.wii = np.loadtxt('./backseat_app/pretrined_agents/wii.txt',delimiter=' ').T
        self.bhi = np.loadtxt('./backseat_app/pretrined_agents/bhi.txt',delimiter=' ').T
        self.bii = np.loadtxt('./backseat_app/pretrined_agents/bii.txt',delimiter=' ').T
        #gt
        self.whg = np.loadtxt('./backseat_app/pretrined_agents/whg.txt',delimiter=' ').T
        self.wig = np.loadtxt('./backseat_app/pretrined_agents/wig.txt',delimiter=' ').T
        self.bhg = np.loadtxt('./backseat_app/pretrined_agents/bhg.txt',delimiter=' ').T
        self.big = np.loadtxt('./backseat_app/pretrined_agents/big.txt',delimiter=' ').T
        #ot
        self.who = np.loadtxt('./backseat_app/pretrined_agents/who.txt',delimiter=' ').T
        self.wio = np.loadtxt('./backseat_app/pretrined_agents/wio.txt',delimiter=' ').T
        self.bho = np.loadtxt('./backseat_app/pretrined_agents/bho.txt',delimiter=' ').T
        self.bio = np.loadtxt('./backseat_app/pretrined_agents/bio.txt',delimiter=' ').T
        #Layer2
        #ft2
        self.whf2 = np.loadtxt('./backseat_app/pretrined_agents/whf2.txt',delimiter=' ').T
        self.wif2 = np.loadtxt('./backseat_app/pretrined_agents/wif2.txt',delimiter=' ').T
        self.bhf2 = np.loadtxt('./backseat_app/pretrined_agents/bhf2.txt',delimiter=' ').T
        self.bif2 = np.loadtxt('./backseat_app/pretrined_agents/bif2.txt',delimiter=' ').T
        #it2
        self.whi2 = np.loadtxt('./backseat_app/pretrined_agents/whi2.txt',delimiter=' ').T
        self.wii2 = np.loadtxt('./backseat_app/pretrined_agents/wii2.txt',delimiter=' ').T
        self.bhi2 = np.loadtxt('./backseat_app/pretrined_agents/bhi2.txt',delimiter=' ').T
        self.bii2 = np.loadtxt('./backseat_app/pretrined_agents/bii2.txt',delimiter=' ').T
        #gt2
        self.whg2 = np.loadtxt('./backseat_app/pretrined_agents/whg2.txt',delimiter=' ').T
        self.wig2 = np.loadtxt('./backseat_app/pretrined_agents/wig2.txt',delimiter=' ').T
        self.bhg2 = np.loadtxt('./backseat_app/pretrined_agents/bhg2.txt',delimiter=' ').T
        self.big2 = np.loadtxt('./backseat_app/pretrined_agents/big2.txt',delimiter=' ').T
        #ot2
        self.who2 = np.loadtxt('./backseat_app/pretrined_agents/who2.txt',delimiter=' ').T
        self.wio2 = np.loadtxt('./backseat_app/pretrined_agents/wio2.txt',delimiter=' ').T
        self.bho2 = np.loadtxt('./backseat_app/pretrined_agents/bho2.txt',delimiter=' ').T
        self.bio2 = np.loadtxt('./backseat_app/pretrined_agents/bio2.txt',delimiter=' ').T
        
        #LSTM Hidden states initial values
        self.ht0  = np.zeros(64) #Initial values for RNN
        self.ct0  = np.zeros(64) #Initial values for RNN
        self.ht02 = np.zeros(64) #Initial values for RNN
        self.ct02 = np.zeros(64) #Initial values for RNN

    #Sigmoid function
    def sigmoid(self,x):
        return 1. / (1. + np.exp(-x))
    #Tanh function
    def tanh(self,x):
        return np.tanh(x)
    #activation function
    def np_relu(self,x):
        for i in range(x.size):
            if x[i]<0.:
                x[i] = 0.
        return x
    
    def lstm1(self,x,h,c):
        #layer 1
        # import pdb; pdb.set_trace()
        ft = self.sigmoid(h.dot(self.whf)+self.bhf + x.dot(self.wif)+self.bif)
        it = self.sigmoid(h.dot(self.whi)+self.bhi + x.dot(self.wii)+self.bii)
        gt =    self.tanh(h.dot(self.whg)+self.bhg + x.dot(self.wig)+self.big)
        ot = self.sigmoid(h.dot(self.who)+self.bho + x.dot(self.wio)+self.bio)
        ct = np.multiply(ft,c) + np.multiply(it,gt)
        ht = np.multiply(self.tanh(ct),ot)
        return ht, ct
    
    def lstm2(self,x,h,c):
        #layer 2
        ft = self.sigmoid(h.dot(self.whf2)+self.bhf2 + x.dot(self.wif2)+self.bif2)
        it = self.sigmoid(h.dot(self.whi2)+self.bhi2 + x.dot(self.wii2)+self.bii2)
        gt =    self.tanh(h.dot(self.whg2)+self.bhg2 + x.dot(self.wig2)+self.big2)
        ot = self.sigmoid(h.dot(self.who2)+self.bho2 + x.dot(self.wio2)+self.bio2)
        ct = np.multiply(ft,c) + np.multiply(it,gt)
        ht = np.multiply(self.tanh(ct),ot)
        return ht, ct
        
    def np_forward(self,x):
        
        #layer1 (input)
        prob1 = x.dot(self.w1rnn) + self.b1rnn
        prob1 = self.np_relu(prob1)
        
        #LSTM (hidden)
        ht1,ct1 = self.lstm1(prob1, self.ht0, self.ct0 )
        ht2,ct2 = self.lstm2(ht1,self.ht02,self.ct02)
        
        #layer2 (hidden)
        prob2 = ht2.dot(self.w2rnn) + self.b2rnn
        prob2 = self.np_relu(prob2)
        
        #layer3 (output)
        mean = prob2.dot(self.w3rnn) + self.b3rnn
        
        #copy next values
        self.ht0  = ht1.copy()
        self.ct0  = ct1.copy()
        self.ht02 = ht2.copy()
        self.ct02 = ct2.copy() 
        
        return mean.clip(-1.0,1.0)    

###########################################################################################################
##############################      Main Tracking Class                          ##########################
########################################################################################################### 
class TargetTracking(object):
    def __init__(self, marl_method = 'Ivan2022'):

        self.num_agents = 2
        self.num_targets = 1
        
        #First we load the RL agent
        if marl_method == 'Ivan2022':
            self.target_estimation_method = 'PF' #set to 'PF' for Particle Filter method, otherways the Least Squarre method will be used.
            self.locpfls = Target()
            self.trained_rl_agent = np_rnn_rl_agent() # A H-LSTM-SAC agent trained with configuration: MASACQMIX_lstm_emofish
        elif marl_method == 'Matteo2025':
            #import jax
            #jax.config.update('jax_platform_name', 'cpu')
            #original name = "mappo_rnn_follow_1v1_10min_training_512steps_utracking_1_vs_1_seed0_vmap0_final.safetensors"
            #model_name = "mappo_rnn_1v1.safetensors"
            #original name = "mappo_transformer_follow_from_1v1_landmarkprop25_1024steps_60ksteps_utracking_1_vs_1_seed0_vmap0.safetensors" #Good for 1target and 1agent
            model_name = "mappo_transformer_1v1.safetensors" #Good for 1target and 1agent
            #original name ="mappo_transformer_tracking_from_1024steps_to_larger_team_utracking_3_vs_1_step24412_rng928981903.safetensors" #Good for 1target and multiple agents
            #model_name = "mappo_transformer_3v1.safetensors" #Good for 1target and multiple agents
            #original name = "mappo_transformer_from_5v5follow_256steps_utracking_5_vs_5_step7320_rng202567368.safetensors"
            #model_name = "mappo_transformer_5v5.safetensors"
            project_root = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            project_root = os.path.abspath(project_root) 
            model_path = os.path.join(project_root, "backseat_app","jaxtorchagent", "IROS_MODELS", model_name)
            self.agent_controller = load(
                    num_agents=self.num_agents-1,
                    num_landmarks=self.num_targets,
                    model_path=model_path,
                    dt=30 # seconds per step
                )
            self.agent_controller.reset(seed=1)
            self.discrete_action_mapping = np.array([-0.24, -0.12, 0, 0.12, 0.24])
        else:
            print('ERROR: Unknown MARL method for target tracking. It should be Ivan2022 or Matteo2025. Exiting')
            exit(-1)

        #set parameters
        self.marl_method = marl_method
        self.last_measureTimestamp = 0 #we need to initialize this variable
        self.lrauv_position = np.array([0.,0.,0.,0.])
        self.agents_pos = np.zeros([self.num_agents,3])
        self.agents_range = np.zeros([self.num_targets,self.num_agents])
        self.lrauv_position_origin = np.array([0.,0.,0.,0.])
        self.lrauv_target_origin = np.array([0.,0.,0.,0.])
        self.initialized = False
        self.zonenumber = 0
        self.zoneletter = 0
        self.lrauvHeading  = 0
        self.ping_count = 0

        #for saving .txt purposes
        folder_name = './logs/'
        if not os.path.isdir(folder_name):
            print('Making foler '+folder_name)
        timestamp = time.time()
        tstr = time.gmtime(timestamp)
        fileName = time.strftime('%Y%m%dT%H%M%S_log.txt', tstr)
        self.fileDirName = folder_name + fileName 
        with open(self.fileDirName,'w') as csvfile:
            np.savetxt(csvfile,[],delimiter=',',header='chan,time,lonWG,latWG,' \
                       + 'slantRange,lonTarg,latTarg,lrauvHeading,planarRange,lrauvDepth')
        print('Log file name: ',self.fileDirName)
        
    
    def newAction(self,targetAddr,agents_slantRange,agents_lrauvLatLon,agents_lrauvDepth,agents_measureTimestamp,new_range=True):
        ''' Track specified targets
        TODO
        ''' 
        #initialize variables to 0s
        self.agents_pos = np.zeros([self.num_agents,3])
        self.agents_range = np.zeros([self.num_targets,self.num_agents])

        #take current lrauv vehicle from all data
        slantRange = np.array(agents_slantRange).item(0)
        lrauvLatLon = agents_lrauvLatLon[0]
        lrauvDepth = agents_lrauvDepth[0]
        measureTimestamp = np.array(agents_measureTimestamp).item(0)
        for i in range(len(agents_slantRange[0])): #TODO: we assume that we have only one target, if more, we need to change the script
            self.agents_range[0][i] = agents_slantRange[0][i] 

        #Compute the planar range based on LRAUV depth and target depth
        #TODO
        planarRange = slantRange + 0.
        
        #If this is the first time since initializon, we save the current LRAUV position
        #as origin
        if self.initialized == False:
            #LRAUV current position in UTM format
            try:
                tuple = utm.from_latlon(lrauvLatLon[0], lrauvLatLon[1])
                lrauv_x, lrauv_y, self.zonenumber, self.zoneletter = tuple
                lrauv_vx = 0.
                lrauv_vy = 0.
                self.lrauv_position_origin = np.array([lrauv_x, lrauv_vx, lrauv_y, lrauv_vy])
                #save the LRAUV origin as the first target estimation position
                self.targetLat = lrauvLatLon[0]
                self.targetLon = lrauvLatLon[1]
                #set the flag
                self.initialized = True
            except:
                print('ERROR: Cannot convert LRAUV Lat/Lon to UTM. Check that the LRAUV Lat/Lon is correct')
                return(-1)


        #compute the position of the current LRAUV in UTM (using the format for Ivan)
        tuple = utm.from_latlon(lrauvLatLon[0], lrauvLatLon[1])
        lrauv_x, lrauv_y, zonenumber, zoneletter = tuple
        lrauv_x -= self.lrauv_position_origin.item(0)
        lrauv_y -= self.lrauv_position_origin.item(2)
        # save the current lrauv postion and velocity
        elapsed_time = measureTimestamp-self.last_measureTimestamp
        if elapsed_time <= 0:
            elapsed_time = 1.
        lrauv_vx = (lrauv_x - self.lrauv_position.item(0))/elapsed_time
        lrauv_vy = (lrauv_y - self.lrauv_position.item(2))/elapsed_time
        # self.lrauv_position = np.array([lrauv_x, lrauv_vx, lrauv_y, lrauv_vy])
        self.lrauv_position = np.array([lrauv_x, lrauv_vx, lrauv_y, lrauv_vy]) 
        self.last_measureTimestamp = measureTimestamp + 0.

        #compute the position of the others LRAUVs in UTM (using the format for Matteo)
        for i in range(len(agents_lrauvLatLon)):
            if agents_lrauvLatLon[i][0] == 0:
                  continue
            tuple = utm.from_latlon(agents_lrauvLatLon[i][0], agents_lrauvLatLon[i][1])
            lrauv_x, lrauv_y, zonenumber, zoneletter = tuple
            lrauv_x -= self.lrauv_position_origin.item(0)
            lrauv_y -= self.lrauv_position_origin.item(2)
            self.agents_pos[i] = np.array([lrauv_x, lrauv_y, agents_lrauvDepth[i]]) 
        
        #if this is the first iteration, we don't go further and it's used only to update the lrauv position
        if self.ping_count == 0:
            self.ping_count += 1
            return(-1,0)

        if self.marl_method == 'Ivan2022':

            print('INFO: Running Ivan2022 MARL method')
            print('INFO: LRAUV pos (x,y,depth)= %.2fm, %.2fm, %.2fm'%(self.lrauv_position[0],self.lrauv_position[2],lrauvDepth))
            print('INFO: Target range= %.2fm'%planarRange)
            print('INFO: MYOBSERVER (x,vx,y,vy)', self.lrauv_position)

            #Update the etimated target position using PF or LS
            if self.target_estimation_method == 'PF':
                #Update the estimated target position using a Particle Filter 
                self.locpfls.updatePF(dt=elapsed_time, new_range=new_range, z=planarRange, myobserver=self.lrauv_position, update=new_range)
            else:
                #Update the estimated target position using a Particle Filter
                self.locpfls.updateLS(dt=elapsed_time, \
                                            new_range=new_range, z=planarRange, \
                                            myobserver=self.lrauv_position)
            #convert back to lat/lon
            self.targetLat, self.targetLon = utm.to_latlon(self.locpfls.position[0]+self.lrauv_position_origin[0], \
                                        self.locpfls.position[2]+self.lrauv_position_origin[2], \
                                        self.zonenumber, self.zoneletter)
            print('INFO: Target_predictions(x,y)=%.2fm,%.2fm'%(self.locpfls.position[0],self.locpfls.position[2]))
        
        elif self.marl_method == 'Matteo2025':
            '''#update estimated target position and new heading using a MARL transformer-based agent
            #angle = 0.0 # radians
            #ranges = np.array([[10.0]]) # (targets, agents), first is always the current agent
            #positions = np.array([[0.0, 0.0, 0.0]]) # (agents, 3), first is always the current agent
            #targets_depth = np.array([10]) # (targets,)'''

            #In Matteo's method a 0 means no measurement.
            self.agents_range = np.where(self.agents_range == -1, 0, self.agents_range)
            #lrauv heading angle (yaw) +-180 degrees, East reference
            angle = np.arctan2(self.lrauv_position[3],self.lrauv_position[1])
            #print('angle arctan=',angle*180./np.pi)
            #Convert yaw to 0-360 degrees, East reference
            if angle <= 0.:
                angle = np.pi+(np.pi+angle)
            #print('angle 360   =',angle*180./np.pi)
            #Add 90 degrees offset to rotate yaw to North reference
            angle = angle-(np.pi/2.)
            angle = angle% (2.*np.pi)
            #print('angle 360  N=',angle*180./np.pi)
            # Convert yaw back to +-180 degrees, with North reference
            if angle > np.pi:
                angle = angle - 2.*np.pi
            #we change the sign to make clockwise positive angles
            angle = -angle +0.
            #print('angle+-180 N=',angle*180./np.pi)
            #Now back to 360 N but with clockwise positive angles
            angle = angle%(2.*np.pi)
            #print('angle 360Ncl=',angle*180./np.pi)
            #ranges = np.array([[planarRange]]) # (targets, agents), first is always the current agent
            #positions = np.array([[self.lrauv_position [0], self.lrauv_position [2], 0.]]) # (agents, 3), first is always the current agent
            targets_depth = np.array([10.]) # (targets,), a constant, not used
            
            print('INFO: Running Matteo2025 MARL method')
            #print('INFO: LRAUV pos (x,y,depth,yaw)= %.2fm, %.2fm, %.2fm, %.2fdegrees'%(self.lrauv_position[0],self.lrauv_position[2],lrauvDepth,angle*180./np.pi))
            #print('INFO: Target range= %.2fm'%planarRange)
            #print('INFO: MYOBSERVER (x,vx,y,vy) ', self.lrauv_position)
            print('INFO: AGENTS_POS (x,y,z) ', self.agents_pos)
            print('INFO: AGENTS_RANGE', self.agents_range)

            self.action, self.target_predictions = self.agent_controller.get_action_and_predictions(
                        angle=angle,
                        ranges=self.agents_range, # (targets, agents), first is always the current agent
                        positions=self.agents_pos,
                        targets_depth=targets_depth,
                        dt=30 # seconds per step
                    )
            print('INFO: Action=',self.action,' target_predictions(x,y)=%.2fm,%.2fm'%(self.target_predictions['landmark_0_tracking_x'],self.target_predictions['landmark_0_tracking_y']))
            
            #convert back to lat/lon
            self.targetLat, self.targetLon = utm.to_latlon(self.target_predictions['landmark_0_tracking_x']+self.lrauv_position_origin[0], \
                                        self.target_predictions['landmark_0_tracking_y'] +self.lrauv_position_origin[2], \
                                        self.zonenumber, self.zoneletter)
    
        #Start stalking. If a new target estimation has not been conducted, 
        #it will use either the LRAUV origin position or the last target estimation if exist
        print('Stalk target on addr ' + str(targetAddr))
        if new_range==True:
            self.ping_count += 1
            self.lrauv_target_origin = self.lrauv_position.copy()
            local_range = planarRange + 0.
        elif self.marl_method == 'Ivan2022':
            #we compute the local range between the LRAUV and the latest estimated target position to feed the RL
            local_range = np.sqrt((self.locpfls.position[0] - self.lrauv_position[0])**2 + (self.locpfls.position[2] - self.lrauv_position[2])**2)

        if self.marl_method == 'Ivan2022':
            #convert to xy the latest estimated target position
            tuple = utm.from_latlon(self.targetLat, self.targetLon)
            t_x, t_y, self.zonenumber, self.zoneletter = tuple
            current_yaw = np.arctan2(self.lrauv_position[3],self.lrauv_position[1])
            obs_fake_speed = [float(np.cos(current_yaw))*0.3,
                                float(np.sin(current_yaw))*0.3,
                                float(self.lrauv_position[0])/1000.,
                                float(self.lrauv_position[2])/1000.,
                                float(t_x-self.lrauv_position_origin[0]-self.lrauv_position[0])/1000., 
                                float(t_y-self.lrauv_position_origin[2]-self.lrauv_position[2])/1000., 
                                float(local_range)/1000.,
                        float(0)/1000., #TODO we should add the real target depth wrt the LRAUV here
                        float(self.lrauv_target_origin[0])/1000.,
                        float(self.lrauv_target_origin[2])/1000.] 
            #print('INFO: Running Ivan2022 RL method')
            #print('INFO: Observation space = ',np.array(obs_fake_speed))
            #0.03 is the speed used to train the model
            #action = self.trained_rl_agent.next_action(obs)
            action = self.trained_rl_agent.np_forward(np.array(obs_fake_speed))
            inc_angle = action * 0.3 #we multiply by 0.3 to limit the minimum angle that the AUV can do

            #print('INFO: lrauv pos = ',self.lrauv_position)
            #print('INFO: Inc angle Action = ',inc_angle*180/np.pi)
            self.lrauvHeading  = self.lrauvHeading + inc_angle
            if self.lrauvHeading  > np.pi*2.:
                self.lrauvHeading -= np.pi*2.
            if self.lrauvHeading  < -np.pi*2:
                self.lrauvHeading  += np.pi*2.
            #print('INFO: Next Heading command to the LRRAUV is %.1f degrees'%((np.pi/2.-self.lrauvHeading)*180/np.pi))   

        elif self.marl_method == 'Matteo2025':
            inc_angle = self.discrete_action_mapping[self.action] #radians
            self.lrauvHeading  = inc_angle+0.
        
        #Save information
        timestamp = time.time()
        tstr = time.gmtime(timestamp)
        print('ODSS format: targAddr,time,lon0,lat0,range,' \
              + 'lonTarg,latTarg,timestring')
        print('RESULT: ' + str(targetAddr) + ',' + \
            str(int(timestamp)) + ',' + \
            str(round(lrauvLatLon[1], 5)) + ',' + \
            str(round(lrauvLatLon[0], 5)) + ',' + \
            str(round(slantRange, 2)) + ',' +  \
            str(round(self.targetLon, 5)) + ',' +  \
            str(round(self.targetLat, 5)) \
            + ',' + time.strftime('%Y-%m-%dT%H:%M:%S', tstr))
        print('INFO: Ping_count='+str(self.ping_count))
        print('')
        #save info into a .txt file                
        #header='chan,time,lonWG,latWG,range,lonTarg,latTarg,lrauvHeading,planarRange,lrauvDepth'
        aux_t=np.array([[targetAddr,timestamp,lrauvLatLon[1],lrauvLatLon[0],slantRange,\
                        self.targetLon,self.targetLat,self.lrauvHeading,\
                        planarRange, lrauvDepth]])                
        with open(self.fileDirName,'a') as csvfile:
            np.savetxt(csvfile,aux_t,delimiter=',')

        #TODO: If the lrauv current possition is too far away from origin, we update origin with current values
        #aux_dist = np.sqrt([(lrauv_x-self.agents_pos[0][0])**2+(lrauv_y-self.agents_pos[0][2])**2])
        #print("agents0pos",self.agents_pos[0])
        #print("auxdist=",aux_dist)
        #if aux_dist < 900 and agents_lrauvLatLon[0][0] != 0: # we set the threshold at 900 m
        #    print("TRUEEEEE")
        #    self.lrauv_position_origin = self.agents_pos[0].copy()
        
        if self.marl_method == 'Ivan2022':
            return((np.pi/2.-self.lrauvHeading)*180/np.pi, aux_t) #we adjust as the Gazebo sim has the North as 0 degrees.
        elif self.marl_method == 'Matteo2025':
            return((self.lrauvHeading)*180/np.pi, aux_t) #we dont need to adjust as in Matteo's method 0 degrees is North
            
    
###########################################################################################################


        
