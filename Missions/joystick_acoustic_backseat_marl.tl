mission joystick_acoustic_backseat_marl {
  """
  This mission is a test to deploy MARL in the Gazebo sim for multi-agent target traccking using range-only.
  """

  arguments {
    # Mission-wide settings

    MissionTimeout = 4 hour
      """
      Maximum duration of mission
      """

    NeedCommsTime = 4 hour
      """
      Elapsed time after previous surface communications when vehicle will
      begin to ascend for additional surface communications
      """

    # You probably need to change these.

    ContactLabel = 9 count
      """
      The acoustic address of the asset to be tracked
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

    SpeedUpdateTimeout = 60 second
      """
      Max time allowed between speed command updates.
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

    # You probably do not need to change these.

    TrackingUpdatePeriod = 15 second
      """
      How long to wait between acoustic queries
      """

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

    YoYoMinDepth = 20 meter
      """
      Minimum depth during yo-yo's.
      """

    YoYoMaxDepth = 30 meter
      """
      Maximum depth during yo-yo's.
      """

    YoYoMinAltitude = 10 meter
      """
      Min altitude while in yo-yo mode.
      """

    YoYoPitch = 20 degree
      """
      Pitch (plus and minus) for yo-yo behavior.
      """

    Speed = 0 meter_per_second
      """
      Vehicle speed.
      """

    # You are even less likely to need to change these.

    KwpHeading = 0.010 radian_per_meter
      """
      Used to relax waypoint cross-track error constant that is adjusted for
      docking. (You can override this setting by passing an argument.)
      """

    MinAltitude = 7 meter
      """
      Minimum height above the sea floor for the entire mission.
      """

    MaxDepth = 105 meter
      """
      Maximum depth for the entire mission.
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


  aggregate UpdateYoYoMaxDepth {
    run when (
      elapsed ( customUri "_.YoYoMaxDepth" ) < ( elapsed ( YoYoMaxDepth ) )
      and ( customUri "_.YoYoMaxDepth" < MaxDepth )
    )

    assign in sequence YoYoMaxDepth = customUri "_.YoYoMaxDepth"

    syslog info "Upadating YoYoMaxDepth " + customUri "_.YoYoMaxDepth"~meter
  }

  aggregate UpdateYoYoMinDepth {
    run when (
      elapsed ( customUri "_.YoYoMinDepth" ) < ( elapsed ( YoYoMinDepth ) )
      and ( customUri "_.YoYoMinDepth" < MaxDepth )
    )

    assign in sequence YoYoMinDepth = customUri "_.YoYoMinDepth"

    syslog info "Upadating YoYoMinDepth " + customUri "_.YoYoMinDepth"~meter
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
    )

    assign in sequence Speed = customUri "_.speedCmd"

    syslog info "Updating vehicle speed " + Speed~meter_per_second
  }

  aggregate UpdateCommandMode {
    run when (
      elapsed ( customUri "_.horizontalCmdMode" ) < ( elapsed ( HorizontalCommandMode ) )
      and (
        customUri "_.horizontalCmdMode" == HeadingMode
        or ( customUri "_.horizontalCmdMode" == RudderMode )
      )
    )

    assign in sequence HorizontalCommandMode = customUri "_.horizontalCmdMode"

    syslog info "Updating horizontal command mode " + HorizontalCommandMode~count
  }

  aggregate UpdateHeading {
    run when (
      elapsed ( customUri "_.headingCmd" ) < ( elapsed ( Heading ) )
    )

    assign in sequence Heading = customUri "_.headingCmd"

    syslog info "Updating heading " + Heading~degree
  }

  aggregate UpdateRudder {
    run when (
      elapsed ( customUri "_.rudderAngleCmd" ) < ( elapsed ( RudderAngle ) )
    )

    assign in sequence RudderAngle = customUri "_.rudderAngleCmd"

    syslog info "Updating rudder angle " + RudderAngle~degree
  }

  # <Aggregate Id="UpdateMassPosition">
  #
  #         <When>
  #             <Elapsed><CustomUri Uri="_.massPositionCmd"/></Elapsed>
  #             <Lt><Elapsed><Arg Name="MassPosition"/></Elapsed></Lt>
  #         </When>
  #
  #         <Assign><Sequence/><Arg Name="MassPosition"/><CustomUri Uri="_.massPositionCmd"/></Assign>
  #
  #         <Syslog Severity="Info">Updating mass position <Arg Name="MassPosition"/><Units:millimeter/></Syslog>
  #
  #     </Aggregate>

   # Power the science payload.

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

  call id="StartingMission" refId="NeedComms"

  # Start mission sequence.

  # <Guidance:Pitch>
  #         <Parallel/>
  #         <Setting><Guidance:Pitch.massPosition/><Control:VerticalControl.massDefault/></Setting>
  #     </Guidance:Pitch>

  behavior Guidance:Buoyancy {
    run in parallel

    set position = Control:VerticalControl.buoyancyNeutral
  }

  behavior Guidance:SetSpeed {
    run in parallel

    set speed = Speed
  }

  aggregate YoYos {
    run in parallel

    behavior Guidance:DepthEnvelope {
      run in parallel

      set minDepth = YoYoMinDepth
      set maxDepth = YoYoMaxDepth
      set pitch = YoYoPitch
    }

    behavior Guidance:AltitudeEnvelope {
      run in parallel

      set minAltitude = YoYoMinAltitude
      set pitch = YoYoPitch
    }

    behavior Guidance:YoYo {
      run in parallel

      set pitch = YoYoPitch
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
