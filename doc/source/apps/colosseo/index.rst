Colosseo
========

.. highlight:: sh

Use :mod:`pyrocko.gf` Greens' Function databases to generate earthquake data from picturebook scenarios.

``colosseo`` is a CLI for :mod:`pyrocko.scenario` which orchestrates different generators to randomly (yet seeded) create synthetic data:

:class:`~pyrocko.scenario.sources.SourceGenerators`
* :mod:`~pyrocko.scenario.sources`

TargetGenerators

* :class:`~pyrocko.scenario.targets.station.StationGenerator`
* :class:`~pyrocko.scenario.targets.waveform.WaveformGenerator`
* :class:`~pyrocko.scenario.targets.insar.InSARGenerator`
* :class:`~pyrocko.scenario.targets.gnss_campaign.GNSSCampaignGenerator`

Creating a scenario with ``colosseo`` is straight forward, but if you want more help with the subcommands, use:

.. code-block :: sh

    colosseo --help


Initialize a new scenario
--------------------------

Create a new scenario in folder :file:`my_scenario`:

.. code-block :: sh

    colosseo init my_scenario


The scenario is built from a YAML configuration file.

.. code-block:: yaml
    :caption: Example scenario configuration file

    --- !pf.scenario.ScenarioGenerator
    avoid_water: true
    center_lat: 52
    center_lon: 5.4
    radius: 90000.0
    ntries: 500
    target_generators:
    - !pf.scenario.RandomStationGenerator
      avoid_water: true
      ntries: 500
      nstations: 8
    - !pf.scenario.WaveformGenerator
      avoid_water: true
      ntries: 500
      station_generator: !pf.scenario.RandomStationGenerator
        avoid_water: true
        ntries: 500
        nstations: 10
      noise_generator: !pf.scenario.WhiteNoiseGenerator
        scale: 1.0e-06
      store_id: crust2_m5_hardtop_8Hz_fine
      seismogram_quantity: displacement
      vmin_cut: 2000.0
      vmax_cut: 8000.0
      fmin: 0.01
    - !pf.scenario.InSARGenerator
      avoid_water: true
      ntries: 500
      store_id: ak135_static
      inclination: 98.2
      apogee: 693000.0
      swath_width: 20000.0
      track_length: 15000.0
      incident_angle: 29.1
      resolution: [250, 250]
      mask_water: true
      noise_generator: !pf.scenario.AtmosphericNoiseGenerator
        amplitude: 1.0
    - !pf.scenario.GNSSCampaignGenerator
      avoid_water: true
      ntries: 500
      station_generator: !pf.scenario.RandomStationGenerator
        avoid_water: true
        ntries: 500
        nstations: 10
      noise_generator: !pf.scenario.GPSNoiseGenerator
        measurement_duarion_days: 2.0
      store_id: ak135_static
    source_generator: !pf.scenario.DCSourceGenerator
      ntries: 500
      avoid_water: false
      nevents: 2
      radius: 1000
      time_min: 2017-01-01 00:00:00
      time_max: 2017-01-03 00:00:00
      magnitude_min: 4.0
      magnitude_max: 7.0
      depth_min: 5000.0
      depth_max: 10000.0



Start the forward model
---------------------------

Start the forward modelling with:

.. code-block:: sh

    colosseo fill my_scenario


The final scenario
-------------------

The directory structure is divided into subfolders holding the forward modelled data as well as individual files for plots, stations, events and StationXML responses.

.. code-block :: text
    :caption: Colosseo directory structure

    my_scenario/         # this directory hosts the scenario
    |-- scenario.yml     # general settings
    |-- waveforms/       # generated waveforms
    |-- insar/           # Kite InSAR scenes
    |-- map.pdf          # GMT map of the scenario


Along with the output of synthetic data the scenarios' map is plotted

.. figure :: /static/scenario_map.png
  :scale: 80%
  :align: center
  :alt: Synthetic scenario map

  Example of an earthquake scenario located in the Netherland's part of the Lower Rhine Plain.
