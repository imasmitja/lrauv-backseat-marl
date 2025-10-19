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

    Speed = 0 meter_per_second
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

  #readData {
  #  Sensor:AHRS_M2.enableBroadcast
  #}

  #readData {
  #  Sensor:DAT.enableBroadcast
  #}

  # Check for additional instructions first.

  call id="StartingMission" priorityHere=false refId="NeedComms"

  #syslog important "SQS: initial value"+ MyTempVar~none_str

  # Start mission sequence.

  # <Guidance:Pitch>
  #         <Parallel/>
  #         <Setting><Guidance:Pitch.massPosition/><Control:VerticalControl.massDefault/></Setting>
  #     </Guidance:Pitch>

  behavior Guidance:Buoyancy {
    run in parallel

    set position = Control:VerticalControl.buoyancyNeutral
  }

  behavior Guidance:Pitch {
    run while ( MassHold )

    set massPosition = Control:VerticalControl.massDefault
  }

  behavior Guidance:Pitch {
    run in parallel

    set depth = Depth
  }

  behavior Guidance:SetSpeed {
    run in parallel

    set speed = Speed
  }

  aggregate Transit {
    run in parallel

    behavior Guidance:SetSpeed {
      run in parallel

      set speed = Speed
    }
  }

  aggregate JoystickControl {
    run in sequence

    behavior Guidance:Buoyancy {
      run in parallel

      set position = Control:VerticalControl.buoyancyDefault
    }

    behavior Guidance:SetSpeed {
      run in parallel

      set speed = Speed
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
