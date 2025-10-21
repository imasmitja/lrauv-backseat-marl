#!/usr/bin/env bash

#
# Copyright (C) Monterey Bay Aquarium Research Institute (MBARI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

errcho() { printf "%s\n" "$*" >&2; }
CMD_X_OPTION=""

#if [ $# -ge 1 ]; then
#  VEHICLE=$1
#  if [ $# -ge 2 ]; then
#      CMD_X_OPTION="-x '$2'"
#  fi
#else
#  echo ""
#  echo "A known vehicle name is required to run the LRAUV application:"
#  echo ""
#  echo "      usage example1: `basename $0` tethys"
#  echo "      usage example2: `basename $0` tethys 'run Science/profile_station.xml'"
#  echo ""
#  exit
#fi


# Vehicles used during the simulation:
#VEHICLE=("brizo" "daphne" "galene" "makai" "pontus" "tethys" "triton")
VEHICLE_agents=("tethys" "pontus")
VEHICLE_targets=("daphne")

# Mission used for each vehicle agent:
CMD_X_OPTION_agent="Engineering/marl.tl"
#CMD_X_OPTION_agent="Engineering/marl2.tl"
#CMD_X_OPTION_agent=""

# Mission used for each vehicle target:
#CMD_X_OPTION_target="-x 'run Default.xml'"
CMD_X_OPTION_target=""


CONTAINER_NAME="sandbox"
MBARI_WS="/home/developer/mbari_ws"

#Kill all tmux processes if they exist
echo "Killing all existing tmux sessions..."
tmux kill-server 2>/dev/null || true
pkill -f "tmux" 2>/dev/null || true
# Clear tmux resurrect files (if plugin exists)
rm -f ~/.tmux/resurrect/* 2>/dev/null || true
# Wait for cleanup
sleep 2

#Launch the Gazebo simulation environment in a tmux windows
tmux new -d -s gazebo-sim
# Check if container is running
if docker ps -q -f "name=^${CONTAINER_NAME}$" | grep -q .; then
        echo '✅ Container is running. Attaching...'
        tmux send-keys -t gazebo-sim.0 "cd ../lrauv-application/" ENTER
        tmux send-keys -t gazebo-sim.0 "./reenter" ENTER
else
        echo '❌ Container is not running. Starting...'
        tmux send-keys -t gazebo-sim.0 "cd ../lrauv-application/" ENTER
        tmux send-keys -t gazebo-sim.0 "./enter" ENTER
        echo '✅ Container is now running. Attaching...'
        
fi
tmux send-keys -t gazebo-sim.0 "pkill -f gz" ENTER #First we need to stop any running gz.
tmux send-keys -t gazebo-sim.0 "gz sim -r ~/mbari_ws/Tools/gz/worlds/portuguese_ledge.sdf" ENTER


# Wait for Gazebo to fully start
echo "Waiting 15 seconds for Gazebo to initialize..."
sleep 15
#read -p "Beforre continue, wait until the Gazebo sim has been launched. Then press enter."



#Launch all vehicles as target:
VEHICLE_TRUE=false
for VEHICLE in ${VEHICLE_targets[@]}; do

        if [ $VEHICLE == 'brizo' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.61:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="11"
        fi

        if [ $VEHICLE == 'daphne' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.62:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="8"
        fi

        if [ $VEHICLE == 'galene' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.63:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="9"
        fi

        if [ $VEHICLE == 'makai' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.64:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="5"
        fi

        if [ $VEHICLE == 'pontus' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.65:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="10"
        fi

        if [ $VEHICLE == 'triton' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.66:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="12"
        fi

        if [ $VEHICLE == 'tethys' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1'"
                VEHICLE_TRUE=true
                TARGET_ID="6"
        fi

        
	
        if VEHICLE_TRUE=true; then
                tmux new -d -s target-$VEHICLE
                tmux send-keys -t target-$VEHICLE.0 "export $LCM_URL" ENTER
                tmux send-keys -t target-$VEHICLE.0 "cd ../lrauv-application/" ENTER
                tmux send-keys -t target-$VEHICLE.0 "./reenter" ENTER
                tmux send-keys -t target-$VEHICLE.0  "cd $MBARI_WS/vehicles" ENTER
                #tmux send-keys -t target-$VEHICLE.0 "./run_lrauv $VEHICLE 'run $CMD_X_OPTION_target'" ENTER
                tmux send-keys -t target-$VEHICLE.0 "$LCM_URL ./run_lrauv $VEHICLE" ENTER
                echo "✅ Launched LRAUV application instance for $VEHICLE."
                sleep 5 #Give it some time to start properly before sending the command
                tmux send-keys -t target-$VEHICLE.0 "run $CMD_X_OPTION_target" ENTER
                echo "✅ Launched LRAUV mission instance for $VEHICLE."
        else
                echo ""
                errcho "ERROR: Failed to start LRAUV application instance for $VEHICLE."
                errcho "Aborting." && exit
        fi
done

#Launch all vehciles as agents:
sleep 2 #Give it some time to start properly before starting agents

VEHICLE_TRUE=false
for VEHICLE in ${VEHICLE_agents[@]}; do

        if [ $VEHICLE == 'brizo' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.61:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'daphne' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.62:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'galene' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.63:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'makai' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.64:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'pontus' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.65:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'triton' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.66:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi

        if [ $VEHICLE == 'tethys' ]; then
                LCM_URL="LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1'"
                VEHICLE_TRUE=true
                
        fi


	if VEHICLE_TRUE=true; then
                tmux new -d -s agent-$VEHICLE
                tmux send-keys -t agent-$VEHICLE.0 "export $LCM_URL" ENTER
                tmux send-keys -t agent-$VEHICLE.0 "cd ../lrauv-application/" ENTER
                tmux send-keys -t agent-$VEHICLE.0 "./reenter" ENTER
                tmux send-keys -t agent-$VEHICLE.0  "cd $MBARI_WS/vehicles" ENTER
                #tmux send-keys -t agent-$VEHICLE.0 "./run_lrauv $VEHICLE 'run $CMD_X_OPTION_agent'" ENTER
                tmux send-keys -t agent-$VEHICLE.0 "$LCM_URL ./run_lrauv $VEHICLE" ENTER
                echo "✅ Launched LRAUV application instance for $VEHICLE."
                sleep 5 #Give it some time to start properly before sending the command
                tmux send-keys -t agent-$VEHICLE.0 "load $CMD_X_OPTION_agent" ENTER
                sleep 4 #Give it some time to start properly before sending the command
                tmux send-keys -t agent-$VEHICLE.0 "set marl.ContactLabel $TARGET_ID count" ENTER

                # Get all other vehicles
                OTHER_VEHICLES=()
                for tempvar in "${VEHICLE_agents[@]}"; do
                        if [[ "$tempvar" != "$VEHICLE" ]]; then
                        OTHER_VEHICLES+=("$tempvar")
                        fi
                done

                for agents in "${OTHER_VEHICLES[@]}"; do

                        if [ $agents == 'brizo' ]; then
                                OTHER_AGENT="11"
                        fi

                        if [ $agents == 'daphne' ]; then
                                OTHER_AGENT="8"
                        fi

                        if [ $agents == 'galene' ]; then
                                OTHER_AGENT="9"
                        fi

                        if [ $agents == 'makai' ]; then
                                OTHER_AGENT="5"
                        fi

                        if [ $agents == 'pontus' ]; then
                                OTHER_AGENT="10"
                        fi

                        if [ $agents == 'triton' ]; then
                                OTHER_AGENT="12"
                        fi

                        if [ $agents == 'tethys' ]; then
                                OTHER_AGENT="6"
                        fi
                        sleep 2 #Give it some time to start properly before sending the command
                        tmux send-keys -t agent-$VEHICLE.0 "set marl.SendDataLabel $OTHER_AGENT count" ENTER
                done
                sleep 2 #Give it some time to start properly before sending the command
                tmux send-keys -t agent-$VEHICLE.0 "run" ENTER
                echo "✅ Launched LRAUV mission instance for $VEHICLE."

                #launch different tmux windows for each lrauv-backseat-marl related to an agent
                tmux new -d -s backseat-$VEHICLE
                tmux send-keys -t agent-$VEHICLE.0 "export $LCM_URL" ENTER
                tmux send-keys -t backseat-$VEHICLE.0 "python3.8 ./backseat_app/main.py -c ./backseat_app/config/app_cfg_$VEHICLE.yml" ENTER
                echo "✅ Launched LRAUV backseat instance for $VEHICLE."

        else
                echo ""
                errcho "ERROR: Failed to start LRAUV application instance for $VEHICLE."
                errcho "Aborting." && exit
        fi

	
done









