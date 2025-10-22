# backseat_app/processing.py
import logging
import struct
import numpy as np
from lrauv.LCM.HandlerBase import LcmHandlerBase
from lrauv.LCM.Publisher import LcmPublisher
from backseat_app.utils import TargetTracking
import time

logger = logging.getLogger('backseat_app')


def array_to_hex(arr):
    """Serialize list of floats to an ASCII hex string."""
    # Pack as double precision floats (8 bytes each)
    packed = struct.pack(f'>{len(arr)}d', *arr)
    return packed.hex()

def hex_to_array(hex_str):
    """Deserialize ASCII hex string back into list of floats."""
    data = bytes.fromhex(hex_str)
    n = len(data) // 8
    return list(struct.unpack(f'>{n}d', data))

def process_other_obs(var):
    """Process the observation state from other vehicles"""
    #This is a placeholder function, you can add any processing you want to do with the data
    #TODO
    #print ("Processing other vehicle observation state: ", var)
    return var

class MarlProcessor(LcmHandlerBase):
    def __init__(self, lcm_instance, cfg):
        #super().__init__(lcm_instance, cfg)
        super().__init__()
        self.publisher = LcmPublisher(lcm_instance)
        self.cfg = cfg
        logger.info('Initializing backseat process (pilot lrauv based on marl)')
        
        #Define and initialate local variables
        self.counter = 0 #used to increment the speed and angle in Joystick mission test
        self.speed_limit = 1.
        self.rudder_limit = 15
        self.heading = 0
        self.rudder = 0
        self.target_timestamp_old = 0
        self.lrauv_pose = [0,0] #[lat,lon]
        self.lrauv_depth = 0
        self.target_range = 0
        self.target_address = 0
        self.target_timestamp = 0
        self.target_timestamp_bsc = time.time()
        self.marl_method = 'Matteo2025'
        #self.marl_method = 'Ivan2022'
        self.rl_tracking = TargetTracking(marl_method = self.marl_method)
        self.new_action = 0
        self.command = "$SR"
        self.speed = 0
        self.latlon_estimation = False
        self.obs_to_send  = []
        self.other_obs_history = np.array([[0,0,0,0,0,0]])
        self.var_to_send = 0.0 #for testing purposes
        self.sim_timestamp = 0.
        self.lastcall = 0.
        self.other_obs_timestamp = 0.
        if self.marl_method == 'Ivan2022':
            self.target_timestamp_bsc_max = 60 #seconds without range measurement before using last information to compute new action
        elif self.marl_method == 'Matteo2025':
            self.target_timestamp_bsc_max = 20 #0.1 #seconds without range measurement before using last information to compute new action

    
    def handle_universal_msg(self, channel, data):
        """Process universal messages"""
        #logger.debug(f"Handling LCM msg on channel {channel}")
        #decode the msg
        msg = self.decode(data)
        #get sim timestamp adn convert to seconds.
        self.sim_timestamp = msg.epochMillisec/1000.
        #get variables names inside the channel msg
        variable_names = self.get_item_names(msg)
        #work with the varaibles we want
        for name in variable_names:
            #Firts we use the Lat/Lon estimated using dead reckogning if it is available, if not
            #we use the regular Lat/Lon from the GPS. This is not necessary, but there is something
            #with the lat/lon that it is not published after a wile...
            #try:
            #    #aux = variable_names.index('horizontal_path_length_since_last_six')
            #    if self.get_variable('latitude',msg).data[0] != 0:
            #        self.latlon_estimation = True
            #        if name == 'latitude':
            #            self.lrauv_pose[0] = self.get_variable(name,msg).data[0]
            #        elif name == 'longitude':
            #            self.lrauv_pose[1] = self.get_variable(name,msg).data[0]
            #except:
            #    if self.latlon_estimation == False:

            if name == 'latitude':
                self.lrauv_pose[0] = self.get_variable(name,msg).data[0]
            elif name == 'longitude':
                self.lrauv_pose[1] = self.get_variable(name,msg).data[0] 

            #Get information from nearby vehicles
            if name == 'others_observations':    
                try:
                    other_obs_value = hex_to_array(self.get_variable(name,msg).data[0])
                except:
                    other_obs_value = self.get_variable(name,msg).data[0]
                other_obs= process_other_obs(other_obs_value)
                if other_obs[0] != self.other_obs_timestamp:
                    self.other_obs_timestamp = other_obs[0]
                    self.other_obs_history = np.array([other_obs_value])
                    logger.info("MARL: Received Other Vehicles Observation State: "+str(other_obs))
                    print("RECEIVED: New data from nearby agents [timestamp, agent address, agent x, agent y, agent z, range]: " + str(other_obs))
                
            #Get depth and acoustic contact information
            if name == 'depth':
                self.lrauv_depth = self.get_variable(name,msg).data[0]
            elif name == 'acoustic_contact_range':
                self.target_range = self.get_variable(name,msg).data[0]
            elif name == 'acoustic_contact_address':
                self.target_address = self.get_variable(name,msg).data[0]
            elif name == 'acoustic_receive_time':
                self.target_timestamp = self.get_variable(name,msg).data[0]

        return


    def compute_new_heading(self):
        """Compute new heading based on agent position, target position, and other agents observation states"""
       
        if self.lrauv_pose[0] == 0:
            self.command = "$SR"
            self.speed = 0.
            return
        #reset the other agents history if it is too old (10 minutes)
        if abs(float(self.other_obs_timestamp)-self.sim_timestamp) > 600:
            self.other_obs_history = np.array([[0,0,0,0,0,0]])
        
        #print("LRAUV pose [%.6f,%.6f,%.2f]: "%(self.lrauv_pose[0], self.lrauv_pose[1],self.lrauv_depth))
        if self.target_timestamp != self.target_timestamp_old: #new range measurement
            print("########################################")   
            print("New range measurement at ",self.target_timestamp)
            print('INFO: Elapsed time = %.3f seconds'%(self.target_timestamp-self.target_timestamp_old))
            print('INFO: Sim timestamp ' + time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(self.sim_timestamp)))
            print("Timestamp = ", self.sim_timestamp)
            print("LRAUV pose [%.6f,%.6f,%.2f]: "%(self.lrauv_pose[0], self.lrauv_pose[1],self.lrauv_depth))
            print("Target address %i at %.3f meters"%(self.target_address,self.target_range))
            logger.debug("New range measured")
            self.target_timestamp_old = self.target_timestamp+0
            agents_timestamp = [self.target_timestamp] + [obs[0] for obs in self.other_obs_history]
            agents_pose = [self.lrauv_pose] + [obs[2:4] for obs in self.other_obs_history]
            agents_depth = [self.lrauv_depth] + [obs[4] for obs in self.other_obs_history]
            agents_range = [[self.target_range] + [obs[5] for obs in self.other_obs_history]]
            self.new_action, internal_state  = self.rl_tracking.newAction(self.target_address,agents_range,agents_pose,agents_depth,agents_timestamp)
            #after we have used the nother observation history to update the PF and take a new acction, we reset it
            self.other_obs_history = np.array([[0,0,0,0,0,0]])
            #TODO: We need to find how to deal when there is more than one target! For now, it works only with one.
            # [Timestamp, lrauv address, lrauv x, lrauv y, lrauv z, range]
            self.obs_to_send = np.array([self.target_timestamp, int(self.cfg['vehicle_id_log']),self.lrauv_pose[0], self.lrauv_pose[1], self.lrauv_depth, self.target_range])
            #publish it to nearby vehicles
            self.publish_observation_state_to_slate()
            #log internal states, actions, and observations
            logger.debug('MARL INFO: internal_state, '+str(internal_state))
            logger.debug('MARL INFO: new_action, '+str(self.new_action))
            #set internal values to control the vehicle
            if self.marl_method == 'Matteo2025':
                print("NEW RUDDER POSITION=",self.new_action)
                if self.new_action != -1:
                    self.new_action = self.new_action + 0.
                    #set command to rudder control and speed
                    self.command = "$SR"
                    self.speed = 1.
                else:
                    self.command = "$SR"
                    self.speed = 0.
            elif self.marl_method == 'Ivan2022':
                print("NEW HEADING=",self.new_action)
                if self.new_action != -1:
                    self.new_action = self.new_action + 0.
                    #set command to heading control and speed
                    self.command = "$SH"
                    self.speed = 1.
                else:
                    self.command = "$SH"
                    self.speed = 0.
            

        elif (self.sim_timestamp - self.target_timestamp_bsc) > self.target_timestamp_bsc_max and self.lrauv_pose[0] != 0 and self.lrauv_pose[1] != 0: #no range measurement for a while
            print("***************************************")
            print("WARNING: No range measurement for a while, using last informaiton to compute new heading")
            print("LRAUV pose [%.6f,%.6f,%.2f]: "%(self.lrauv_pose[0], self.lrauv_pose[1],self.lrauv_depth))
            #Compute elapsed time since last call
            print('INFO: Elapsed time = %.3f seconds'%(self.sim_timestamp-self.target_timestamp_bsc) )
            print('INFO: Sim timestamp ' + time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(self.sim_timestamp)))
            logger.debug("WARNING: No range measurement for a while, using last informaiton to compute new heading")
            self.target_timestamp_bsc = self.sim_timestamp+0.
            agents_timestamp = [self.target_timestamp] + [obs[0] for obs in self.other_obs_history]
            agents_pose = [self.lrauv_pose] + [obs[2:4] for obs in self.other_obs_history]
            agents_depth = [self.lrauv_depth] + [obs[4] for obs in self.other_obs_history]
            agents_range = [[-1] + [obs[5] for obs in self.other_obs_history]]
            if np.array(agents_range).sum() != -1:
                aux_range = True
            else:
                aux_range = False
            self.new_action, internal_state = self.rl_tracking.newAction(self.target_address,agents_range,agents_pose,agents_depth,agents_timestamp,new_range=aux_range)
            #after we have used the nother observation history to update the PF and take a new acction, we reset it
            self.other_obs_history = np.array([[0,0,0,0,0,0]])
            #log internal states, actions, and observations
            logger.debug('MARL INFO: internal_state, '+str(internal_state))
            logger.debug('MARL INFO: new_action, '+str(self.new_action))
            if self.marl_method == 'Matteo2025':
                #print("NEW RUDDER POSITION=",self.new_action)
                if self.new_action != -1:
                    self.new_action = self.new_action + 0.
                    #set command to rudder control and speed
                    self.command = "$SR"
                    self.speed = 1.
                else:
                    self.command = "$SR"
                    self.speed = 0.
            elif self.marl_method == 'Ivan2022':
                #print("NEW HEADING=",self.new_action)
                if self.new_action != -1:
                    self.new_action = self.new_action + 0.
                    #set command to heading control and speed
                    self.command = "$SH"
                    self.speed = 1.
                else:
                    self.command = "$SH"
                    self.speed = 0.
     
        return
        

    def publish_data_to_slate(self, channel_name='tethys_slate'):
        """
        TODO: UPDATE WITH YOUR PUBLISHER CODE!

        :param channel_name: LCM channel name (string)
        :return: publishes LCM message
        """

        if self.command == "$SR":
            self.speed = min(float(self.speed),self.speed_limit)
            #self.rudder = min(float(self.new_action),self.rudder_limit)
            self.rudder = np.clip(self.new_action,-self.rudder_limit,self.rudder_limit)
                        
            msg = "$SR," + str(self.speed) + ',' + str(self.rudder) + ';'

            self.publisher.add_int("_.horizontalCmdMode", 1, "count")
            self.publisher.add_float("_.speedCmd", self.speed, "m/s")
            self.publisher.add_float("_.rudderAngleCmd", -self.rudder, "degree")
            self.publisher.publish(self.cfg["lcm_data_pub_channel"])

        elif self.command == "$SH":
            #self.speed = min(float(data[1]), self.speed_limit)
            #self.heading = (self.heading + float(data[2])) % 360
            self.heading = self.new_action % 360
            
            self.publisher.add_int("_.horizontalCmdMode", 0, "count")
            self.publisher.add_float("_.speedCmd", self.speed, "m/s")
            self.publisher.add_float("_.headingCmd", self.heading, "degree")
            self.publisher.publish(self.cfg["lcm_data_pub_channel"])
                        
            msg = "$SH," + str(self.speed) + ',' + str(self.heading) + ';'


    def publish_observation_state_to_slate(self, channel_name='tethys_slate'):
        """Publish a detection alert to the vehicle's slate"""
        #Compressing observation state to be send
        aux = array_to_hex(self.obs_to_send)
        #this is only for testing purposes
        if int(self.cfg['vehicle_id_log']) == 6:
                self.var_to_send = 610
                print('INFO: Sending current observation of vehicle address 6 to addres 10 (x,y,z,range)',self.obs_to_send)
                #print('In HEX: ',aux)
        elif int(self.cfg['vehicle_id_log']) == 10:
                self.var_to_send = 106
                print('INFO: Sending current observation of vehicle address 10 to addres 6 (x,y,z,range)',self.obs_to_send)
                #print('In HEX: ',aux)
        
        self.publisher.clear_msg()
        # publish LCM message
        self.publisher.add_variable(
            name='_.send_observations',
            val=aux,
            unit='none_str'
        )
        self.publisher.publish(channel_name)

        return


