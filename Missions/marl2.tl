mission marl {
  """
  This mission is a test to deploy MARL in the Gazebo sim for multi-agent target traccking using range-only.
  """

  arguments {
    # Mission-wide settings

    MissionTimeout = 4 hour
      """
      Maximum duration of mission
      """

    NeedCommsTime = 1 hour
      """
      Elapsed time after previous surface communications when vehicle will
      begin to ascend for additional surface communications
      """

    # You probably need to change these.

    ContactLabel = 8 count
      """
      The acoustic address of the asset to be tracked. This is the default value, but it will be overwritten by bash file duirng simulation launch.
      tethys = 6
      brizo = 11
      daphne = 8
      galene = 9
      makai = 5
      pontus = 10
      triton = 12
      """

    SendDataLabel = 10 count
      """
      The acoustic address of the asset to send observation data 255 broadcast (it doesn't work,says it is not mapped), 10 is pontus 
      This is the default value, but it will be overwritten by bash file duirng simulation launch.
      """
    
    ContactDepth = NaN meter
      """
      The asset's depth to be tracked
      """

    CenterLatitude = 36.81544 degree
      """
      The latitude of the center of the circle. Fill this in to specify a
      start position, it will be overwritten once the vehicle receives an
      acoustic signal from the contact.
      """

    CenterLongitude = -121.82506 degree
      """
      The longitude of the center of the circle. Fill this in to specify a
      start position, it will be overwritten once the vehicle receives an
      acoustic signal from the contact.
      """

    CenterNorthingsDelta = 0 meter
      """
      Desired change in latitude to achieve, expressed as a distance.
      """

    CenterEastingsDelta = 0 meter
      """
      Desired change in longitude to achieve, expressed as a distance.
      """

    CircleRadius = 50 meter
      """
      Radius of circle around acoustic contact during wait time
      """

    SpeedUpdateTimeout = 240 second
      """
      Max time allowed between speed command updates.
      """
    StateUpdateDelay = 10 second
      """
      Minimum time delay between state updates to avoid rapid changes.
      """

    HorizontalCommandMode = 0 count
      """
      Mission horizontal command mode. 0 = Heading command (default). 1 = Rudder angle command.
      """

    AcousticContactTimeout = 4 hour
      """"
        If the vehicle does not receive an acoustic signal for more than this
        length of time, it will surface for communications with shore.
      """

    SendObservationDataInterval = 1 minute
      """
      How often to send back some a-comms data
      """

    TrackingUpdatePeriod = 30 second
      """
      How long to wait between acoustic queries
      """

        # parameters for constant depth trajectory
    Depth = 20 meter
      """
      Depth of flat and level flight during the mission.
      """

    MaxDepth = 60 meter
      """
      Maximum depth for the entire mission.
      """
    MinAltitude = 5 meter
      """
      Minimum height above the sea floor for the entire mission.
      """
    MassHold = true
      """
      Set to True in order to hold mass at default position, False to allow
      mass to run on its own.
      """

    BuoyancyHold = true
      """
      Set to True in order to hold buoyancy at neutral position, False to
      allow buoyancy to run on its own.
      """

    Latitude = NaN degree
      """
      Latitude of waypoint to hold. If NaN, hold the latitude at start of
      mission.
      """

    Longitude = NaN degree
      """
      Longitude of waypoint to hold. If NaN, hold the longitude at start of
      mission.
      """

    SinkOffSurface = false
      """
      If true, use buoyancy only to decend to SinkDepth before driving to first target depth
      """

    SinkDepthTimeout = 10 minute
      """
      Max wait time for the vehicle to reach the target depth
      """

      #  You probably do not need to change these.
    


    NumberOfPings = 1 count
      """
      Number of return pings to request with each acoustic query (more than 1
      will activate oneway mode)
      """

    CircleMaxError = 10 meter
      """
      If this distance away from the circle, drive straight towards (or away
      from the center). Otherwise, try to reduce distance from the ideal
      circle.
      """

    CircleTurnToPort = false
      """
      If true, vehicle turns to the left around the center point. If false,
      vehicle turns to the right.
      """

    Speed = 1 meter_per_second
      """
      Vehicle speed.
      """

    MyTempVar = "init"

      """
      A custom variable  from the backseat driver interface.
      """
    # You are even less likely to need to change these.

    KwpHeading = 0.010 radian_per_meter
      """
      Used to relax waypoint cross-track error constant that is adjusted for
      docking. (You can override this setting by passing an argument.)
      """

    MinOffshore = 1.5 kilometer
      """
      Minimum offshore distance for the entire mission.
      """

  }

  output {
    HeadingMode = 0 count
      """
      Heading angle command mode enumaration.
      """

    RudderMode = 1 count
      """
      Rudder angle command mode enumaration.
      """

    Heading = 0 degree
      """
      Commanded vehicle heading. Mission will init Heading to current vehicle heading.
      """

    RudderAngle = 0 degree
      """
      Commanded vehicle RudderAngle. Only acitve when mission is in rudder commands mode.
      """

    TimeLastSendData = 0 second
    ElapsedSinceLastSendData = 0 second
      """
      Time elapsed since last Send Data fix.
      """ 
    
    DiveMode = 0 count
      """
      Mission variable (don't change). The mission sets this variable to
      switch between dive modes. Initialized to DoingComms.
      """

    DoingComms = 0 count
      """
      Mission variable (don't change). The mission will run in this mode
      when doing surface comms.
      """

    SurfaceSink = 1 count
      """
      Mission variable (don't change). The mission will run in this mode
      when sinking from the surface using only buoyancy.
      """

    TransitToStation = 2 count
      """
      Mission variable (don't change). The mission will run in this mode
      when performing the transit to station
      """

    StationKeep = 3 count
      """
      Mission variable (don't change). The mission will run in this mode
      when keeping station
      """

    OnStation = false
      """
      Mission variable (don't change). Determines last mode (Transit or StationKeep)
      to resume after needcomms
      """
  }

  # Missions should always start with a timeout.

  timeout duration=MissionTimeout

  # Add tracking directive high in the stack (above NeedComms to enable tracking from surface).

  insert Insert/AbortDrift.tl {
    redefineArg AcousticTimeout = AcousticContactTimeout
  }

  behavior Estimation:TrackAcousticContact {
    run in parallel

    set contactDepthSetting = ContactDepth
    set contactLabelSetting = ContactLabel
    set numberOfSamplesSetting = NumberOfPings
    set updatePeriodSetting = TrackingUpdatePeriod
  }

  #assign in sequence customUri "_.others_observations" = "999"

  aggregate SendObservationData {

     run when ( elapsed (MyTempVar) >= SendObservationDataInterval )

     syslog info "Got test_good : " + customUri "_.send_observations"~none_str + " and temp var is " + MyTempVar~none_str
     assign in sequence ElapsedSinceLastSendData = elapsed (MyTempVar) 

     assign in sequence MyTempVar = customUri "_.send_observations"

     behavior Sensor:SendDirect {
        run in parallel

        set destType = "modem"
        set destId = SendDataLabel
        set destName = "_.others_observations"
        set value = MyTempVar
        set unit = "none_str"

     }
     syslog info "SendData: sent send_observations : " + MyTempVar~none_str + " to nearby vehicles. Elapsed time " + ElapsedSinceLastSendData~minute

  }


  aggregate UpdateContactLabel {
    run when (
      elapsed ( customUri "_.contactLabelSetting" ) < ( elapsed ( ContactLabel ) )
    )

    assign in sequence ContactLabel = abs ( customUri "_.contactLabelSetting" )

    syslog info "Upadating ContactLabel " + customUri "_.contactLabelSetting"~count
  }

  # Most missions have the NeedComms aggregate at high priority.

  insert id="NeedComms" Insert/NeedComms.tl

  assign in sequence NeedComms:DiveInterval = NeedCommsTime


  # Missions should almost always start with standard safety envelopes; most missions should not expose the parameters of these envelopes.

  insert Insert/StandardEnvelopes.tl

  assign in sequence StandardEnvelopes:MinAltitude = MinAltitude

  assign in sequence StandardEnvelopes:MaxDepth = MaxDepth

  assign in sequence StandardEnvelopes:MinOffshore = MinOffshore

  # Power the science payload.
  #
  #    <Insert Filename="Insert/Science.tl" Id="Science">
  #        <Description>
  #            Get science data, including PeakDetectChl
  #        </Description>
  #        <RedefineArg Name="PeakDetectChlActive"><True/></RedefineArg>
  #    </Insert>

  # Add guidance update directives high in the stack.

  aggregate SpeedTimeout {
    run when ( elapsed ( Speed ) >= SpeedUpdateTimeout )

    assign in sequence Speed = 0 meter_per_second

    syslog fault "Speed timeout reached. Stopping vehicle (speed " + Speed~meter_per_second + ")."
  }

  aggregate UpdateSpeed {
    run when (
      elapsed ( customUri "_.speedCmd" ) < ( elapsed ( Speed ) )
      and (elapsed ( Speed) > StateUpdateDelay)
    )

    assign in sequence Speed = customUri "_.speedCmd"

    #syslog info "Updating vehicle speed " + Speed~meter_per_second
  }

  aggregate UpdateCommandMode {
    run when (
      elapsed ( customUri "_.horizontalCmdMode" ) < ( elapsed ( HorizontalCommandMode ) )
      and (
        customUri "_.horizontalCmdMode" == HeadingMode
        or ( customUri "_.horizontalCmdMode" == RudderMode )
      )
      and (elapsed ( HorizontalCommandMode) > StateUpdateDelay)
    )

    assign in sequence HorizontalCommandMode = customUri "_.horizontalCmdMode"

    #syslog info "Updating horizontal command mode " + HorizontalCommandMode~count
  }

  aggregate UpdateHeading {
    run when (
      elapsed ( customUri "_.headingCmd" ) < ( elapsed ( Heading ) )
      and (elapsed ( Heading) > StateUpdateDelay)
    )

    assign in sequence Heading = customUri "_.headingCmd"

    syslog info "Updating heading " + Heading~degree
  }

  aggregate UpdateRudder {
    run when (
      elapsed ( customUri "_.rudderAngleCmd" ) < ( elapsed ( RudderAngle ) )
      and (elapsed ( RudderAngle) > StateUpdateDelay)
    )

    assign in sequence RudderAngle = customUri "_.rudderAngleCmd"

    syslog info "Updating rudder angle " + RudderAngle~degree
  }


 

  insert id="Science" Insert/Science.tl

  # Power the Backseat payload.

  insert Insert/BackseatDriver.tl

  assign in sequence BackseatDriver:EnableBackseat = true

  insert Insert/PowerOnly.tl

  # Power the Backseat payload.

  behavior Guidance:BackseatDriver {
    run in parallel
  }



  # Start mission sequence.


  aggregate SurfaceComms {
    run when (
      called
      or ( elapsed ( Universal:time_fix ) > NeedCommsTime )
    )

    assign in sequence DiveMode = DoingComms

    call priorityHere=false refId="NeedComms"

    aggregate HandleNan {
      run in sequence

      break if (
        not isNaN ( Latitude )
        and not isNaN ( Longitude )
      )

      # If both lat and lon are nan, we are implicitly on station and should skip initial transit, so force OnStation to true
      aggregate SetOnStation {
        run in sequence

        break if (
          not isNaN ( Latitude )
          or not isNaN ( Longitude )
        )

        assign in sequence OnStation = false #we dont use this
      }

      # The "assigns" here mean we should only ever run these aggregates once
      # An initial NaN lat/lon should be set here during the initial surface NeedComms call
      # Use the most recent fix -- if there isn't one, the keepstation behavior will use Universal:lat/lon
      aggregate SetLat {
        run in sequence

        break if ( not isNaN ( Latitude ) )

        assign in sequence Latitude = Universal:latitude_fix
      }

      aggregate SetLon {
        run in sequence

        break if ( not isNaN ( Longitude ) )

        assign in sequence Longitude = Universal:longitude_fix
      }
    }

    aggregate setTransit {
      run in sequence

      break if (
        SinkOffSurface
        or OnStation
      )

      assign in sequence DiveMode = TransitToStation
    }


    aggregate setSink {
      run in sequence

      break if ( not SinkOffSurface )

      assign in sequence DiveMode = SurfaceSink
    }
  }

  call id="StartingMission" priorityHere=false refId="SurfaceComms"

  

  aggregate DiveCmd {
    run when ( called )

    aggregate SurfaceSink {
      run while ( DiveMode == SurfaceSink )

      aggregate sink {
        run in sequence

        behavior Guidance:Buoyancy id="BuoyancyHold" {
          run in parallel

          set position = NaN cc
        }

        behavior Guidance:SetSpeed {
          run in parallel

          set speed = 0 m/s
        }

        syslog info "Sinking to " + Depth~meter

        behavior Guidance:Pitch {
          run in sequence

          timeout duration=SinkDepthTimeout {
            syslog important "Timed out while trying to reach target sink depth at current depth of "
                 + Universal:depth~meter
          }

          set depth = Depth
        }

        # Once we've sunk, either resume transit or station keeping depths and speeds
        aggregate sinkToTransit {
          run in sequence

          break if ( OnStation )

          assign in sequence DiveMode = TransitToStation
        }
      }
    }

    aggregate TransitToStation {
      run while ( DiveMode == TransitToStation )

      aggregate dive {
        run in sequence

        behavior Guidance:Buoyancy {
          run in parallel

          set position = Control:VerticalControl.buoyancyNeutral
        }

        behavior Guidance:Pitch {
          run in parallel

          set depth = Depth
        }

        behavior Guidance:SetSpeed {
          run in parallel

          set speed = Speed
        }
      }
    }
  }


  aggregate JoystickControl {
    """
    This separete aggregate is to modify on the fly the rudder or heading based on the marl algorithm
    that is running on the backseat computer.
    """
    run in sequence

    aggregate Dive {
      run while ( not ( DiveMode == DoingComms ) )

      call priorityHere=false refId="DiveCmd"
    }


    aggregate CmdHeading {
      run while ( HorizontalCommandMode == HeadingMode )

      behavior Guidance:Point {
        run in parallel

        set forceUpdate = true
        set heading = Heading
      }
    }

    aggregate CmdRudder {
      run while ( HorizontalCommandMode == RudderMode )

      behavior Guidance:Point {
        run in parallel

        set forceUpdate = true
        set rudderAngle = RudderAngle
      }
    }
  }
}

