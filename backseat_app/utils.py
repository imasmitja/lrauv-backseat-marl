# -*- coding: utf-8 -*-
"""
Created on 2025

@author: Ivan Masmitja Rusinol

Project: MARL Fulbright CSIC-MBARI
"""

import numpy as np
import random
import time
import utm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from backseat_app.jaxtorchagent.production_agent import load


###########################################################################################################
##############################      Main Tracking Class                          ##########################
########################################################################################################### 
class TargetTracking(object):
    def __init__(self):

        self.num_agents = 2
        self.num_targets = 1
        
        #1- First we load the RL agent
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


        #2- set parameters
        self.last_measureTimestamp = 0 #we need to initialize this variable
        self.lrauv_position = np.array([0.,0.,0.,0.])
        self.agents_pos = np.zeros([self.num_agents,3])
        self.agents_range = np.zeros([self.num_targets,self.num_agents])
        self.lrauv_position_origin = np.array([0.,0.,0.,0.])
        self.initialized = False
        self.zonenumber = 0
        self.zoneletter = 0
        self.lrauvAction  = 0
        self.ping_count = 0

        #3 for saving .txt purposes
        folder_name = './logs/'
        if not os.path.isdir(folder_name):
            print('Making foler '+folder_name)
        timestamp = time.time()
        tstr = time.gmtime(timestamp)
        fileName = time.strftime('%Y%m%dT%H%M%S_log.txt', tstr)
        self.fileDirName = folder_name + fileName 
        with open(self.fileDirName,'w') as csvfile:
            #header='chan,time,lonWG,latWG,range,lonTarg,latTarg,lrauvAction,planarRange,lrauvDepth,agents_pos(all),agentts_range(all),origin'
            np.savetxt(csvfile,[],delimiter=',',header='chan,time,lonWG,latWG,' \
                       + 'slantRange,lonTarg,latTarg,lrauvAction,planarRange,lrauvDepth,'\
                        +str(['agent_'+str(i)+'_pos(x,y,z)' for i in range(self.num_agents)])\
                            +str(['agent_'+str(n)+'_target_'+str(i)+'_range' for i in range(self.num_targets) for n in range(self.num_agents)]) \
                                + ',origin(x,vx,y,vy)')
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
        
        #Next:
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
        #print('INFO: Running Matteo2025 MARL method')
        #print('INFO: LRAUV pos (x,y,depth,yaw)= %.2fm, %.2fm, %.2fm, %.2fdegrees'%(self.lrauv_position[0],self.lrauv_position[2],lrauvDepth,angle*180./np.pi))
        #print('INFO: Target range= %.2fm'%planarRange)
        #print('INFO: MYOBSERVER (x,vx,y,vy) ', self.lrauv_position)
        print('INFO: AGENTS_POS (x,y,z) ', self.agents_pos)
        print('INFO: AGENTS_RANGE', self.agents_range)
        #update target prediciton and obtain new action at once
        self.action, self.target_predictions = self.agent_controller.get_action_and_predictions(
                    angle=angle,
                    ranges=self.agents_range, # (targets, agents), first is always the current agent
                    positions=self.agents_pos,
                    targets_depth=targets_depth,
                    dt=30 # seconds per step
                )
        print('INFO: Action=',self.action,' target_predictions(x,y)=%.2fm,%.2fm'%(self.target_predictions['landmark_0_tracking_x'],self.target_predictions['landmark_0_tracking_y']))

        #convert the action from MARL agent into rudder action:
        inc_angle = self.discrete_action_mapping[self.action] #radians
        self.lrauvAction  = inc_angle+0.
        
        #convert back to lat/lon
        self.targetLat, self.targetLon = utm.to_latlon(self.target_predictions['landmark_0_tracking_x']+self.lrauv_position_origin[0], \
                                    self.target_predictions['landmark_0_tracking_y'] +self.lrauv_position_origin[2], \
                                    self.zonenumber, self.zoneletter)
    
        #Start stalking. If a new target estimation has not been conducted, 
        #it will use either the LRAUV origin position or the last target estimation if exist
        print('Stalk target on addr ' + str(targetAddr))
        if new_range==True:
            self.ping_count += 1
            local_range = planarRange + 0.
        
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
        #header='chan,time,lonWG,latWG,range,lonTarg,latTarg,lrauvAction,planarRange,lrauvDepth,agents_pos(all),agentts_range(all),origin'
        aux_t=np.array([targetAddr,\
                        timestamp,\
                        lrauvLatLon[1],\
                        lrauvLatLon[0],\
                        slantRange,\
                        self.targetLon,\
                        self.targetLat,\
                        self.lrauvAction,\
                        planarRange,\
                        lrauvDepth,\
                        ])  
        aux_t = np.concatenate([aux_t,\
                        self.agents_pos.flatten(),\
                        self.agents_range.flatten(),\
                        self.lrauv_position_origin.flatten(),\
                        ])    
        aux_t = np.matrix(aux_t)
        with open(self.fileDirName,'a') as csvfile:
            np.savetxt(csvfile,aux_t,delimiter=',')

        #If the lrauv current possition is too far away from origin, and we are close to the target, we update origin with current target position
        aux_lrauv_dist = np.sqrt([(self.lrauv_position[0])**2+(self.lrauv_position[2])**2])
        aux_target_dist = np.sqrt([(self.lrauv_position[0]-self.target_predictions['landmark_0_tracking_x'])**2+(self.lrauv_position[2]-self.target_predictions['landmark_0_tracking_y'])**2])
        ## we set the lrauv distance threshold at 900 m and the target distance to 400.
        if aux_lrauv_dist > 500 and aux_target_dist < 200 and agents_lrauvLatLon[0][0] != 0:
            print('')
            print("******************************************************************************") 
            print("WARNING: Updating ORIGIN POSSITION with current target position. LRAUV distance from origin is %.3f m, and LRAUV-TARGET distance is %.3f"%(aux_lrauv_dist, aux_target_dist))
            #Set the new lrauv_position_origin variable
            aux_x = self.target_predictions['landmark_0_tracking_x']+self.lrauv_position_origin[0]
            aux_y = self.target_predictions['landmark_0_tracking_y']+self.lrauv_position_origin[2]
            self.lrauv_position_origin = np.array([aux_x,0.,aux_y,0.])
            print("WARNING: New origin possition set to "+str(self.lrauv_position_origin))
            #Reset MARL networks
            print("WARNING: reseting internal MARL values")
            self.agent_controller.reset(seed=1)
            #Reset the PF using new origin as initial point
            print('WARNING: reseting PF trakcing')
            print("******************************************************************************") 
            print('')
            for i, tracker in enumerate(self.agent_controller.trackers):
                tracker.model.init_particles(position=np.array([0.,0.,0.,0.]), slantrange=100)

        
        return((self.lrauvAction)*180/np.pi, aux_t) #we dont need to adjust as in Matteo's method 0 degrees is North
            
    
###########################################################################################################


        
