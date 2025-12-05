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



try:
    from backseat_app.jaxtorchagent.production_agent import load
except:
    from jaxtorchagent.production_agent import load



###########################################################################################################
##############################      Main Tracking Class                          ##########################
########################################################################################################### 
class TargetTracking(object):
    def __init__(self, agent_version='1v1'):

        if agent_version == "1v1":        
            #1- First we load the RL agent
            #This is the agent we tested on Octover 2025 and worked great 1v1
            model_name = "mappo_transformer_1v1_v4.safetensors" # New agent with should work: mappo_transformer_noisy_more_linear_utracking_1_vs_1_step456_rng1948878966
            self.num_agents = 1
            self.num_targets = 1
            print('INFO: Using AGENT VERSION 1 agent vs 1 target (aka 1v1)')
        elif agent_version == "2v1":
            model_name = "mappo_2v1_2december.safetensors" #Same agent as previouse but trained for 2v1
            self.num_agents = 2
            self.num_targets = 1
            print('INFO: Using AGENT VERSION 2 agent vs 1 target (aka 2v1)')
        else:
            print ('ERROR. AGENT VERSION NEED TO BE SPECIFIED CORRECTLY')


        project_root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        project_root = os.path.abspath(project_root) 
        model_path = os.path.join(project_root, "backseat_app","jaxtorchagent", "IROS_MODELS", model_name)
        self.agent_controller = load(
                num_agents=self.num_agents,
                num_landmarks=self.num_targets,
                model_path=model_path,
                dt=30, # seconds per step
            )

        self.agent_controller.reset(seed=1)
        self.discrete_action_mapping = np.array([-0.24, -0.12, 0, 0.12, 0.24])
        self.discrete_action_mapping_heading = np.array([0.903, 0.452, 0.00, -0.452, -0.903])


        #2- set parameters
        self.last_measureTimestamp = 0 #we need to initialize this variable
        self.last_measureTimestamp_reset = 0 #we need to initialize this variable
        self.lrauv_position = np.array([0.,0.,0.,0.])
        self.agents_pos = np.zeros([self.num_agents,3])
        self.agents_range = np.zeros([self.num_targets,self.num_agents])
        self.lrauv_position_origin = np.array([0.,0.,0.,0.])
        self.initialized = False
        self.zonenumber = 0
        self.zoneletter = 0
        self.lrauvAction  = 0
        self.ping_count = 0
        self.agent_range_reset = False

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
        
    
    def newAction(self,targetAddr,agents_slantRange,agents_lrauvLatLon,agents_lrauvDepth,agents_measureTimestamp,new_range=True, action_control = '$SH'):
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
        print('DEBUG: agents_slantRange=',agents_slantRange)
        print('DEBUG: self.agents_range=',self.agents_range)
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
        
        #Testing to invert y position
        #lrauv_y = -lrauv_y

        # save the current lrauv postion and velocity
        elapsed_time = measureTimestamp-self.last_measureTimestamp
        if elapsed_time <= 0:
            elapsed_time = 1.
        lrauv_vx = (lrauv_x - self.lrauv_position.item(0))/elapsed_time
        lrauv_vy = (lrauv_y - self.lrauv_position.item(2))/elapsed_time
        self.lrauv_position = np.array([lrauv_x, lrauv_vx, lrauv_y, lrauv_vy]) 
        self.last_measureTimestamp = measureTimestamp + 0.

        #compute the position of the Current and Others LRAUVs in UTM (using the format for Matteo)
        for i in range(len(agents_lrauvLatLon)):
            if agents_lrauvLatLon[i][0] == 0:
                  continue
            tuple = utm.from_latlon(agents_lrauvLatLon[i][0], agents_lrauvLatLon[i][1])
            lrauv_x, lrauv_y, zonenumber, zoneletter = tuple
            lrauv_x -= self.lrauv_position_origin.item(0)
            lrauv_y -= self.lrauv_position_origin.item(2)

            #Testing to invert y position
            #lrauv_y = -lrauv_y

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
        #we change the sign to make clockwise positive angles. In the first implementation, we saw that the first Gazebo implementation
        #uses a counterclockwise convention, so we don't need to negate it here.
        angle = -angle +0.
        #print('angle+-180 N=',angle*180./np.pi)
        #Now back to 360 N but with clockwise positive angles
        angle = angle%(2.*np.pi)
        print('Agen yaw (360NorthClockwise)=',angle*180./np.pi)
        #ranges = np.array([[planarRange]]) # (targets, agents), first is always the current agent
        #positions = np.array([[self.lrauv_position [0], self.lrauv_position [2], 0.]]) # (agents, 3), first is always the current agent
        targets_depth = np.array([10.]) # (targets,), a constant, not used
        #print('INFO: Running Matteo2025 MARL method')
        #print('INFO: LRAUV pos (x,y,depth,yaw)= %.2fm, %.2fm, %.2fm, %.2fdegrees'%(self.lrauv_position[0],self.lrauv_position[2],lrauvDepth,angle*180./np.pi))
        #print('INFO: Target range= %.2fm'%planarRange)
        #print('INFO: MYOBSERVER (x,vx,y,vy) ', self.lrauv_position)
        print('INFO: agents range: ',self.agents_range)
        print('INFO: agents pos: ', self.agents_pos)
        print('INFO: targets_depth: ', targets_depth)
        print('INFO: new_range: ', new_range)
        #update target prediciton and obtain new action at once
        self.action, self.target_predictions = self.agent_controller.get_action_and_predictions(
                    ranges=self.agents_range, # (targets, agents), first is always the current agent
                    positions=self.agents_pos,
                    targets_depth=targets_depth,
                    dt=30, # seconds per step
                    new_range=new_range
                )
        print('INFO: Action=',self.action,' target_predictions(x,y)=%.2fm,%.2fm'%(self.target_predictions['landmark_0_tracking_x'],self.target_predictions['landmark_0_tracking_y']))

        if action_control == '$SR':
            #convert the action from MARL agent into rudder action:
            inc_angle = self.discrete_action_mapping[self.action] #radians
            self.lrauvAction  = inc_angle+0.
        else:
            #convert the action from MARL agent into heading action:
            inc_heading = -self.discrete_action_mapping_heading[self.action] #we need to negate the action to make it work as in ruder action
            new_heading = (angle + inc_heading)%(2*np.pi)
            self.lrauvAction = new_heading+0.
        
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

        #####################################################################################
        #########   RESET STATE
        #####################################################################################        
        #If the lrauv current possition is too far away from origin, and we are close to the target, we update origin with current target position
        aux_lrauv_dist = np.sqrt([(self.lrauv_position[0])**2+(self.lrauv_position[2])**2])
        aux_target_dist = np.sqrt([(self.lrauv_position[0]-self.target_predictions['landmark_0_tracking_x'])**2+(self.lrauv_position[2]-self.target_predictions['landmark_0_tracking_y'])**2])
        ## we set the lrauv distance threshold at 900 m and the target distance to 400.
        #if ((aux_lrauv_dist > 80000 and aux_target_dist < 400) or aux_lrauv_dist > 150000) and agents_lrauvLatLon[0][0] != 0:
        if self.agents_range[0][0] > 1000 and self.agent_range_reset == True and agents_lrauvLatLon[0][0] != 0:
            print('')
            print("******************************************************************************") 
            print("WARNING: Updating ORIGIN POSSITION with current target position. LRAUV distance from origin is %.3f m. LRAUV distance from Target is %.3f m"%(aux_lrauv_dist,self.agents_range[0][0]))
            #Set the new lrauv_position_origin variable
            #centered over the target
            #aux_x = self.target_predictions['landmark_0_tracking_x']+self.lrauv_position_origin[0]
            #aux_y = self.target_predictions['landmark_0_tracking_y']+self.lrauv_position_origin[2]
            #centered over the agent
            #aux_x = self.lrauv_position[0]+self.lrauv_position_origin[0]
            #aux_y = self.lrauv_position[2]+self.lrauv_position_origin[2]
            #self.lrauv_position_origin = np.array([aux_x,0.,aux_y,0.])
            #print("WARNING: New origin possition set to "+str(self.lrauv_position_origin))
            #Reset MARL networks
            print("WARNING: reseting internal MARL values")
            self.agent_controller.reset(seed=11)
            #Reset the PF using new origin as initial point
            print('WARNING: reseting PF trakcing')
            print("******************************************************************************") 
            print('')
            self.agent_range_reset = False
            #Enable this if we want to preserve the PF target position that was estimated before the reset
            #for i, tracker in enumerate(self.agent_controller.trackers):
                #centered over the target
                #aux_x = 0.
                #aux_y = 0.
                #centered over the agent
            #    aux_x = self.target_predictions['landmark_0_tracking_x']-self.lrauv_position[0]
            #    aux_y = self.target_predictions['landmark_0_tracking_y']-self.lrauv_position[2]
            #    tracker.model.init_particles(position=np.array([aux_x,0.,aux_y,0.]), slantrange=100, method='area')
            #    tracker.pred[0] = aux_x +0.
            #    tracker.pred[1] = aux_y +0.

        if self.agents_range[0][0] < 500 and self.agents_range[0][0] != 0:
            self.agent_range_reset = True

        reset_time = measureTimestamp-self.last_measureTimestamp_reset 
        reset_time_treshold = 300
        reset_time_flag = False
        print('reset time=',reset_time)
        if reset_time> reset_time_treshold and reset_time_flag == True:
            self.last_measureTimestamp_reset = measureTimestamp
            print('')
            print("******************************************************************************") 
            print("WARNING: LRAUV distance from origin is %.3f m. LRAUV traveled time %.3f min"%(aux_lrauv_dist,reset_time/60.))
            #Reset MARL networks
            print("WARNING: reseting internal MARL values")
            self.agent_controller.actor.reset(seed=10)
            print("******************************************************************************") 
            print('')

        
        return((self.lrauvAction)*180/np.pi, aux_t) #we dont need to adjust as in Matteo's method 0 degrees is North
            
    
###########################################################################################################


        
