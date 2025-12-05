# backseat_app/processing.py
import logging
import struct
import numpy as np
from lrauv.LCM.HandlerBase import LcmHandlerBase
from lrauv.LCM.Publisher import LcmPublisher
import time
import utm

try:
    from backseat_app.utils import TargetTracking
except:
    from utils import TargetTracking

logger = logging.getLogger('backseat_app')

def array_to_hex(arr):
    """Serialize list of floats to an ASCII hex string."""
    """
    Compact encoding with specified precision:
    - timestamp: seconds (4 bytes, up to ~68 years from 1970)
    - address: 0-100 (1 byte unsigned)
    - lat, lon: 6 decimals (4 bytes each, ±0.000001° ≈ 0.11m)
    - depth: 0.1m precision (2 bytes, 0-6553.5m)
    - range: 0.1m precision (2 bytes, 0-6553.5m)
    
    Total: 16 bytes (32 hex chars) vs original 48 bytes (using double  precisio floats everywhere)
    """
    # Unpack array
    timestamp, address, lat, lon, depth, range_meas = arr

    # Pack with optimal precision
    packed = struct.pack(
        '>IBffHH',  # Format: uint32, uint8, float, float, uint16, uint16
        int(timestamp),        # seconds as 32-bit unsigned int. range (0-4294967295)
        int(address),          # address as 8-bit unsigned (0-255)
        round(lat, 6),         # latitude with 6 decimal places
        round(lon, 6),         # longitude with 6 decimal places
        int(round(depth * 10)),  # depth in 0.1m units (e.g., 152.3m = 1523) range (0-65535)
        int(round(range_meas * 10))  # range in 0.1m units range (0-65535)
    )
    return packed.hex()

def hex_to_array(hex_str):
    """Deserialize ASCII hex string back into list of floats."""
    """Decode compact telemetry data"""
    data = bytes.fromhex(hex_str)
    
    # Unpack according to our format
    timestamp, address, lat, lon, depth_encoded, range_encoded = struct.unpack('>IBffHH', data)
    
    return [
        float(timestamp),      # seconds
        float(address),        # address
        round(lat, 6),         # lat
        round(lon, 6),         # lon
        round(depth_encoded / 10.0, 1),  # Convert back to meters
        round(range_encoded / 10.0, 1)   # Convert back to meters
    ]

def process_other_obs(var):
    """Process the observation state from other vehicles"""
    #This is a placeholder function, you can add any processing you want to do with the data
    #TODO
    #print ("Processing other vehicle observation state: ", var)
    return var

def calculate_waypoint(lat, lon, angle_degrees, distance_m=700):
    """
    Calculate waypoint using UTM coordinates for accurate local distance calculations.
    
    Args:
        lat (float): Vehicle latitude in degrees
        lon (float): Vehicle longitude in degrees
        angle_degrees (float): Angle from vehicle in degrees (0° = North, clockwise)
        distance_m (float): Distance to waypoint in meters
    
    Returns:
        tuple: (waypoint_lat, waypoint_lon) in degrees
    """
    
    # Convert to UTM coordinates (easting, northing in meters)
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    
    #print(f"Original UTM - Easting: {easting:.2f}m, Northing: {northing:.2f}m")
    #print(f"UTM Zone: {zone_number}{zone_letter}")
    
    # Convert angle
    angle_rad = np.radians(angle_degrees)
    
    # Calculate displacement in UTM coordinates
    # UTM uses Easting (x-axis, east-west) and Northing (y-axis, north-south)
    delta_easting = distance_m * np.sin(angle_rad)   # East-West component
    delta_northing = distance_m * np.cos(angle_rad)  # North-South component
    
    #print(f"Displacement - Easting: {delta_easting:.2f}m, Northing: {delta_northing:.2f}m")
    
    # Calculate new UTM coordinates
    new_easting = easting + delta_easting
    new_northing = northing + delta_northing
    
    #print(f"New UTM - Easting: {new_easting:.2f}m, Northing: {new_northing:.2f}m")
    
    # Convert back to latitude/longitude
    waypoint_lat, waypoint_lon = utm.to_latlon(new_easting, new_northing, zone_number, zone_letter)
    
    return waypoint_lat, waypoint_lon

# Check for invalid values
def is_valid_coordinate(lat, lon):
    """Check if coordinates are valid and reasonable"""
    
    # Check for NaN, infinity, etc.
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False
    
    # Check reasonable geographic ranges
    if not (-90 <= lat <= 90):
        return False
    
    if not (-180 <= lon <= 180):
        return False
    
    return True

class MarlProcessor(LcmHandlerBase):
    def __init__(self, lcm_instance, cfg):
        #super().__init__(lcm_instance, cfg)
        super().__init__()
        self.publisher = LcmPublisher(lcm_instance)
        self.cfg = cfg
        logger.info('Initializing backseat process (pilot lrauv based on marl)')
        #read the cfg file and set parameters
        self.agent_version = cfg.get("agent_version", "1v1")  # default to '1v1' if not specified
        if self.agent_version == "1v1":
            print("INFO: Loading 1v1 agent version")
            self.num_agents = 1
            self.num_targets = 1
        elif self.agent_version == "2v1":
            print("INFO: Loading 2v1 agent version")
            self.num_agents = 2
            self.num_targets = 1
        else:
            raise ValueError(f"Invalid agent_version: {self.agent_version}. Must be '1v1' or '2v1'.")
        
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
        self.target_range_old = 0
        self.target_address = 0
        self.target_timestamp = 0
        self.target_timestamp_bsc = time.time()
        self.rl_tracking = TargetTracking(self.agent_version)
        self.new_action = 0
        self.command = "$SW"  #set it to: $SH for heading control; $SR for rudder control
        self.speed = 0
        self.latlon_estimation = False
        self.obs_to_send  = []
        if self.num_agents == 1:
            self.other_obs_history = []
        else:
            self.other_obs_history = np.array([[0,0,0,0,0,0]])
        self.var_to_send = 0.0 #for testing purposes
        self.sim_timestamp = 0.
        self.lastcall = 0.
        self.other_obs_timestamp = 0.
        self.target_timestamp_bsc_max = 30 #0.1 #seconds without range measurement before using last information to compute new action
        self.new_action_flag = False
        self.aux_test = True
        self.auxlat = 0
        self.auxlon = 0
        self.target_address_from_mission = 0
        self.other_comms_count = 0

    def handle_universal_msg(self, channel, data):
        """Process universal messages"""
        #logger.debug(f"Handling LCM msg on channel {channel}")
        #decode the msg
        msg = self.decode(data)
        #get sim timestamp adn convert to seconds.
        self.sim_timestamp = msg.epochMillisec/1000.
        #get variables names inside the channel msg
        variable_names = self.get_item_names(msg)
        #print(variable_names)
        #work with the varaibles we want
        for name in variable_names:
            #Firts we use the Lat/Lon estimated using dead reckogning if it is available, if not
            #we use the regular Lat/Lon from the GPS. This is not necessary, but there is something
            #with the lat/lon that it is not published after a while..
            if name == 'latitude':
                self.lrauv_pose[0] = self.get_variable(name,msg).data[0]
            elif name == 'longitude':
                self.lrauv_pose[1] = self.get_variable(name,msg).data[0] 

            #Get information from nearby vehicles
            if name == 'othersObservations':    
                try:
                    other_obs_value = hex_to_array(self.get_variable(name,msg).data[0])
                except:
                    other_obs_value = self.get_variable(name,msg).data[0]
                other_obs= process_other_obs(other_obs_value)
                try:
                    if other_obs[0] != self.other_obs_timestamp:
                        self.other_obs_timestamp = other_obs[0]
                        self.other_obs_history = np.array([other_obs_value])
                        #print('WE SET MANUALLY TO 0 TO SEE PF ON SINGLE VEHICLES')
                        #self.other_obs_history = np.array([[0,0,0,0,0,0]])
                        print('***************************************************************************************************')
                        logger.info("MARL: Received Other Vehicles Observation State: "+str(other_obs))
                        print("RECEIVED: New data from nearby agents [timestamp, agent address, agent x, agent y, agent z, range]: " + str(other_obs))
                        print('***************************************************************************************************')
                        self.other_comms_count +=1
                except:
                    pass
                
            #Get depth and acoustic contact information
            if name == 'depth':
                self.lrauv_depth = self.get_variable(name,msg).data[0]
                if self.lrauv_depth < 0:
                    self.lrauv_depth = 0.
                #print('WARNING, we set manually the depth')
                #self.lrauv_depth  = 0.
            elif name == 'acoustic_contact_range':
                self.target_range = self.get_variable(name,msg).data[0]
            elif name == 'acoustic_contact_address':
                self.target_address = self.get_variable(name,msg).data[0]
            elif name == 'acoustic_receive_time':
                if self.target_address != 1: #TODO. This should be modified dinamically with the address of the target.
                    self.target_timestamp = self.get_variable(name,msg).data[0]
            
            if name=='contactLabelToLcm':
                self.target_address_from_mission = self.get_variable(name,msg).data[0]
                
        return

    def compute_new_heading(self):
        """Compute new heading based on agent position, target position, and other agents observation states"""
       
        if self.lrauv_pose[0] == 0:
            self.speed = 0.75
            return
        #reset the other agents history if it is too old (20 minutes)
        if abs(float(self.other_obs_timestamp)-self.sim_timestamp) > 1200:
            if self.num_agents == 1:
                self.other_obs_history = []
            else:
                self.other_obs_history = np.array([[0,0,0,0,0,0]])
        
        #print("LRAUV pose [%.6f,%.6f,%.2f]: "%(self.lrauv_pose[0], self.lrauv_pose[1],self.lrauv_depth))
        #TODO we set the dat contact address manually to 20 but this shoudl be improve to be more dinamically set based on .cfg or lrauv-app
        if self.target_timestamp != self.target_timestamp_old and self.target_range != self.target_range_old and self.target_address == self.target_address_from_mission and self.target_range != 0 and self.lrauv_pose[0] != 0 and self.lrauv_pose[1] != 0: #new range measurement
            print("########################################")   
            print("New range measurement at ",self.target_timestamp)
            print('INFO: Elapsed time = %.3f seconds'%(self.target_timestamp-self.target_timestamp_old))
            print('INFO: Sim timestamp ' + time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(self.sim_timestamp)))
            print("Timestamp = ", self.sim_timestamp)
            print("LRAUV pose [%.6f,%.6f,%.2f]: "%(self.lrauv_pose[0], self.lrauv_pose[1],self.lrauv_depth))
            print("Target address %i at %.3f meters"%(self.target_address,self.target_range))
            logger.debug("New range measured")
            self.target_timestamp_old = self.target_timestamp+0
            self.target_range_old = self.target_range+0 
            agents_timestamp = [self.target_timestamp] + [obs[0] for obs in self.other_obs_history]
            agents_pose = [self.lrauv_pose] + [obs[2:4] for obs in self.other_obs_history]
            agents_depth = [self.lrauv_depth] + [obs[4] for obs in self.other_obs_history]
            agents_range = [[self.target_range] + [obs[5] for obs in self.other_obs_history]]
            self.new_action, internal_state  = self.rl_tracking.newAction(self.target_address,agents_range,agents_pose,agents_depth,agents_timestamp, new_range=True, action_control=self.command)
            #TODO: We need to find how to deal when there is more than one target! For now, it works only with one.
            # [Timestamp, lrauv address, lrauv x, lrauv y, lrauv z, range]
            self.obs_to_send = np.array([self.target_timestamp, int(self.target_address_from_mission),self.lrauv_pose[0], self.lrauv_pose[1], self.lrauv_depth, self.target_range])
            if self.num_agents > 1:
                #publish it to nearby vehicles
                self.publish_observation_state_to_slate()
                #after we have used the nother observation history to update the PF and take a new acction, we reset it
                self.other_obs_history = np.array([[0,0,0,0,0,0]])
            #log internal states, actions, and observations
            logger.debug('MARL INFO: internal_state, '+str(internal_state))
            logger.debug('MARL INFO: new_action, '+str(self.new_action))
            #set internal values to control the vehicle
            print("NEW RUDDER POSITION=",self.new_action)
            print('INFO: other_comms_count='+str(self.other_comms_count))
            if self.new_action != -1:
                self.new_action = self.new_action + 0.
                #set command to rudder control and speed
                self.speed = 1.
            else:
                self.speed = 0.75
            self.new_action_flag = True

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
            self.new_action, internal_state = self.rl_tracking.newAction(self.target_address,agents_range,agents_pose,agents_depth,agents_timestamp,new_range=aux_range,action_control=self.command)
            #after we have used the other observation history to update the PF and take a new acction, we reset it
            if self.num_agents > 1:
                self.other_obs_history = np.array([[0,0,0,0,0,0]])
            #log internal states, actions, and observations
            logger.debug('MARL INFO: internal_state, '+str(internal_state))
            logger.debug('MARL INFO: new_action, '+str(self.new_action))
            print("NEW RUDDER POSITION=",self.new_action)
            print('INFO: other_comms_count='+str(self.other_comms_count))
            if self.new_action != -1:
                self.new_action = self.new_action + 0.
                #set command to rudder control and speed
                self.speed = 1.
            else:
                self.speed = 0.75
            self.new_action_flag = True

        return
        

    def publish_data_to_slate(self, channel_name='tethys_slate'):
        """
        TODO: UPDATE WITH YOUR PUBLISHER CODE!

        :param channel_name: LCM channel name (string)
        :return: publishes LCM message
        """


        #manula configuration for debug:
        debug_mode = False
        if debug_mode == True:
            self.command = "$SW" #we control the heading
            #self.command = "$SR" #we control the rudder
            self.speed = 1.
            aux_action = 90 #in degrees, from 0 to 360.
            self.new_action = 180
            #aux_action = 6.8 #in degrees, [-13.7, -6.8, 0, 6.8, 13.7]

        if self.command == "$SR":
            self.speed = min(float(self.speed),self.speed_limit)
            #self.rudder = min(float(self.new_action),self.rudder_limit)
            #we observed that the action computed need to be negated to make it work
            aux_action = -self.new_action
            self.rudder = np.clip(aux_action,-self.rudder_limit,self.rudder_limit)
                        
            msg = "$SR," + str(self.speed) + ',' + str(self.rudder) + ';'

            self.publisher.add_int("_.horizontalCmdMode", 1, "count")
            self.publisher.add_float("_.speedCmd", self.speed, "m/s")
            self.publisher.add_float("_.rudderAngleCmd", self.rudder, "degree")
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

        elif self.command == "$SW" and self.new_action_flag == True:
            self.new_action_flag = False
                
            #self.speed = min(float(data[1]), self.speed_limit)
            #self.heading = (self.heading + float(data[2])) % 360
            self.heading = self.new_action % 360

            #compute wp lat/lon based on current vehicle lat/lon and heading
            self.auxlat = self.lrauv_pose[0]
            self.auxlon = self.lrauv_pose[1]
            if is_valid_coordinate(self.auxlat, self.auxlon):
                wplat, wplon = calculate_waypoint(self.auxlat, self.auxlon, self.heading)
            if is_valid_coordinate(wplat, wplon):
                self.publisher.add_int("_.horizontalCmdMode", 2, "count")
                self.publisher.add_float("_.speedCmd", self.speed, "m/s")
                self.publisher.add_float("_.wpLatCmd", wplat, "degree")
                self.publisher.add_float("_.wpLonCmd", wplon, "degree")
                self.publisher.publish(self.cfg["lcm_data_pub_channel"])
                            
                msg = "$SW," + str(self.speed) + ',' + str(wplat) + ',' + str(wplon) + ';'
                print(msg)
                print('lrauv lat', self.auxlat, ' lrauv lon', self.auxlon)


    def publish_observation_state_to_slate(self, channel_name='tethys_slate'):
        """Publish a detection alert to the vehicle's slate"""
        #Compressing observation state to be send
        aux = array_to_hex(self.obs_to_send)
        #this is only for testing purposes
        print('INFO: Sending current observation to other vehicles (Timestamp, address, lat, lon, depth, range)',self.obs_to_send)
        print('INFO: in exa is: ', aux)

        # publish LCM message
        self.publisher.clear_msg()
        self.publisher.add_variable(
            name='_.send_observations',
            val=aux,
            unit='none_str'
        )
        self.publisher.publish(channel_name)

        return


