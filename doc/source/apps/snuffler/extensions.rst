
Extending Snuffler with plugins: Snufflings
===========================================

Snufflings are small Python scripts which extend the functionality of Snuffler.
Snuffler looks into the directory ``$HOME/.snufflings`` for snufflings.
Snufflings can be reloaded at run-time using the menu entry 'Reload Snufflings'
in the main menu of Snuffler - no need to restart Snuffler when a snuffling is
modified or added.

Already existing snufflings can be downloaded from the  `snuffling repository <https://git.pyrocko.org/pyrocko/contrib-snufflings>`_

Example Snuffling to show earthquake catalog information within Snuffler
------------------------------------------------------------------------

Put the following code into ``$HOME/.snufflings/geofon.py``. It will add four
items into the *Snufflings* sub-menu of Snuffler (*Get GEOFON events*, *Get
GEOFON events (> M6)*, ...). When one of these is selected by the user, the
`GEOFON catalog <http://geofon.gfz-potsdam.de/eqinfo/form.php>`_ is queried for
earthquake information for the time range visible in Snuffler. For each
earthquake found, a marker is shown in the viewer.

::

    from pyrocko.snuffling import Snuffling, Param
    from pyrocko.pile_viewer import Marker, EventMarker
    from pyrocko import catalog

    class GeofonEvents(Snuffling):
        
        '''
        Get events from GEOFON catalog.
        '''

        def __init__(self, magmin=None):
            self._magmin = magmin
            Snuffling.__init__(self)

        def setup(self):
            '''Customization of the snuffling.'''
            
            if self._magmin is None:
                self.set_name('Get GEOFON Events')
            else:
                self.set_name('Get GEOFON Events (> M %g)' % self._magmin)
            
        def call(self):
            '''Main work routine of the snuffling.'''
            
            # get time range visible in viewer
            viewer = self.get_viewer()
            tmin, tmax = viewer.get_time_range()
            
            # download event information from GEOFON web page
            # 1) get list of event names
            geofon = catalog.Geofon()
            event_names = geofon.get_event_names(
                time_range=(tmin,tmax), 
                magmin=self._magmin)
                
            # 2) get event information and add a marker in the snuffler window
            for event_name in event_names:
                event = geofon.get_event(event_name)
                marker = EventMarker(event)
                self.add_markers([marker])
                    
    def __snufflings__():
        '''Returns a list of snufflings to be exported by this module.'''
        
        return [ GeofonEvents(), 
                 GeofonEvents(magmin=6), 
                 GeofonEvents(magmin=7), 
                 GeofonEvents(magmin=8) ]

How it works
^^^^^^^^^^^^

Snuffler looks into the directory ``HOME/.snufflings`` for python scripts
(``*.py``). Within each of these it tries to query the function
``__snufflings__()`` which should return a list of snuffling objects, which are
instances of a snuffling class. A custom snuffling class is created by
subclassing :py:class:`pyrocko.snuffling.Snuffling`. Within the derived class implement
the methods ``setup()`` and ``call()``. ``setup()`` is called during
initialization of the snuffling object, ``call()`` is called when the user
selects the menu entry. You may define several snuffling classes within one
snuffling source file. You may also return several instances of a single
snuffling class from the ``__snufflings__()`` function.

The :py:class:`pyrocko.snuffling.Snuffling` base class documentation can also
be accessed with the command ``pydoc pyrocko.snuffling.Snuffling`` from the
shell. Example snufflings can be found in `src/snufflings/ <https://git.pyrocko.org/pyrocko/pyrocko/src/master/src/gui/snufflings>`_
in the pyrocko source code. More examples may be found in the 
`contrib-snufflings repository <https://git.pyrocko.org/pyrocko/contrib-snufflings>`_ repository.

More examples
-------------

Print minimum, maximum, and peak-to-peak amplitudes to the terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a built-in snuffling of Snuffler. It serves here as a demonstration of
how selected trace data can be accessed from within the snuffling.

::

    from pyrocko.snuffling import Snuffling, Param
    from pyrocko import trace

    class MinMaxSnuffling(Snuffling):
        
        '''Reports minimum, maximum, and peak-to-peak values of selected data.
    To use it, use the picker tool to mark a region or select existing regions
    and call this snuffling. The values are printed via standard output to the
    termimal.'''

        def setup(self):
            '''Customization of the snuffling.'''
            
            self.set_name('Minimum Maximum Peak-To-Peak')
            self.tinc = None

        def call(self):
            '''Main work routine of the snuffling.'''
                    
            # to select a reasonable increment for the chopping, the smallest
            # sampling interval in the pile is looked at. this is only done,
            # the first time the snuffling is called.
            if self.tinc is None:
                self.tinc = self.get_pile().get_deltats()[0] * 10000.
            
            # the chopper yields lists of traces but for minmax() below, an iterator
            # yielding single traces is needed; using a converter:
            def iter_single_traces():
                for traces in self.chopper_selected_traces(tinc=self.tinc, 
                                                           degap=False, 
                                                           fallback=True):
                    for tr in traces:
                        yield tr
            
            # the function minmax() in the trace module can get minima and maxima
            # grouped by (network,station,location,channel):
            mima = trace.minmax(iter_single_traces())
            
            for nslc in sorted(mima.keys()):
                p2p = mima[nslc][1] - mima[nslc][0]
                print '%s.%s.%s.%s: %12.5g %12.5g %12.5g' % (nslc + mima[nslc] + (p2p,))
                                                
    def __snufflings__():
        '''Returns a list of snufflings to be exported by this module.'''
        
        return [ MinMaxSnuffling() ]


How to add simple markers to the viewer
---------------------------------------

::

    from pyrocko.snuffling import Snuffling
    from pyrocko.pile_viewer import Marker

    class Example1(Snuffling):
        
        '''Example Snuffling to demonstrate how to add markers to the viewer.

    It looks at all selected traces and puts a Marker at the peak amplitude of the
    raw traces. If no traces are selected all traces in view are used.  It is not
    affected by filter settings of the viewer.

    This works well for short continuous traces, but if longer or gappy traces are
    in the viewer, there may be some problems which are not 
    '''

        def setup(self):
            # this sets the name for the menu entry:
            self.set_name('Example 1: mark peak amplitudes')

        def call(self):
            
            # remove all markers which have been previously added by this snuffling
            self.cleanup()

            # this is a shortcut to get selected traces or all traces in view
            for traces in self.chopper_selected_traces(fallback=True):

                for tr in traces:
                    net, sta, loc, cha = tr.nslc_id

                    # using a trace method to get time and amplitude
                    time, amplitude = tr.absmax()

                    # the marker kind sets the color of the marker
                    kind = 3 

                    # create the marker object
                    m = Marker([ (net, sta, loc, cha) ], time, time, kind )

                    # add it to the viewer
                    self.add_marker(m)

    def __snufflings__():
        return [ Example1() ]

Synthetic Seismograms of an STS2 seismometer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This snuffling demonstrates the method add_paramter() which extends the snufflings' panel by scroll bars and options to choose between predefined parameters. 

::
    
    class STS2:

        ''' Apply the STS2's transfer function which is deduced from the
    poles, zeros and gain of the transfer tunction. The Green's function 
    database (gdfb) which is required for synthetic seismograms and the 
    rake of the focal mechanism can be chosen and changed within snuffler.
    Two gfdbs are needed.
    Three synthetic seismograms of an STS2 seismometer will be the result.
    '''
        # 'evaluate() will apply the transfer function on each frequency.
        def evaluate(self,freqs):

            # transform the frequency to angular frequency.
            w = 2j*pi*freqs

            Poles = array([-3.7e-2+3.7e-2j, -3.7e-2-3.7e-2j,
                           -2.51e2, -1.31e2+4.67e2j, -1.31e2-4.67e2])
            Zeros = array([0,0])
            K = 6.16817e7

            # Multiply factored polynomials of the transfer function's numerator
            # and denominator.
            a = ones(freqs.size,dtype=complex)*K
            for i_z in Zeros:
                a *= w-i_z
            for i_p in Poles:
                a /= w-i_p
            return a

    class ParaEditCp_TF_GTTG(Snuffling):

        def setup(self):

            # Give the snuffling a name:
            self.set_name('STS-2.1')

            # Add scrollbars of the parameters that you desire to adjust.
            # 1st argument: Description that appears within the snuffling.
            # 2nd argument: Name of parameter as used in the following code.
            # 3rd-5th argument: default, start, stop.
            self.add_parameter(Param('Strike[deg]', 'strike', 179., -180., 180.))

            # The parameter 'Choice' adds a menu to choose from different options.
            # 1st argument: Description that appears within the snuffling.
            # 2nd argument: Name of paramter as used in the following code.
            # 3rd argument: Default
            # 4th to ... argument: List containing all other options.
            self.add_parameter(Choice('GFDB','database','gemini',['gemini','qseis']))
            self.set_live_update(False)

        def call(self):

            self.cleanup()

            # Set up receiver configuration.
            tab = '''
            HH  53.456  9.9247  0
            '''.strip()

            receivers = []
            station, lat, lon, depth = tab.split()
            r = receiver.Receiver(lat,lon, components='neu', name='.%s.' % station)
            receivers.append(r)

            # Composition of the source
            olat, olon = 36.9800, -3.5400
            otime = util.str_to_time('1954-03-29 06:16:05')

            # The gfdb can be chosen within snuffler.
            # This refers to the 'add_parameter' method.
            if self.database == 'gemini':
                db = gfdb.Gfdb('/scratch/local2/gfdb_workshop_iasp91/gfdb/db')
            else:
                db = gfdb.Gfdb('/scratch/local2/gfdb_building/deep/gfdb_iasp/db')

            seis = seismosizer.Seismosizer(hosts=['localhost'])
            seis.set_database(db)
            seis.set_effective_dt(db.dt)
            seis.set_local_interpolation('bilinear')
            seis.set_receivers(receivers)
            seis.set_source_location( olat, olon, otime)
            seis.set_source_constraints (0, 0, 0, 0 ,0 ,-1)
            self.seis = seis

            # Change strike within snuffler with the added scroll bar.
            strike = self.strike

            # Other focal mechism parameters are constants
            dip = 122; rake = 80; moment = 7.00e20; depth = 650000; risetime = 24
            s = source.Source('bilateral',
            sourceparams_str='0 0 0 %g %g %g %g %g 0 0 0 0 1 %g' % (depth, moment, strike, dip, rake, risetime))
            self.seis.set_source(s)
            recs = self.seis.get_receivers_snapshot( which_seismograms = ('syn',), which_spectra=(), which_processing='tapered')

            trs = []
            for rec in recs:
                rec.save_traces_mseed(filename_tmpl='%(whichset)s_%(network)s_%(station)s_%(location)s_%(channel)s.mseed' )
                trs.extend(rec.get_traces())

            # Define fade in and out, band pass filter and cut off fader for the TF.
            tfade = 8
            freqlimit = (0.005,0.006,1,1.3)
            cut_off_fading = 5
            ntraces = []

            for tr in trs:
                TF = STS2()

                # Save synthetic trace after transfer function was applied.
                trace_filtered = tr.transfer(tfade, freqlimit, TF, cut_off_fading) 
                # Set new codes to the filtered trace to make it identifiable.
                rename={'e':'BHE','n':'BHN','u':'BHZ'}
                trace_filtered.set_codes(channel=rename[trace_filtered.channel], network='', station='HHHA', location='syn')
                ntraces.append(trace_filtered)

    #             Extract the synthetic trace's data with get_?data() and store them.
    #            xval = trace_filtered.get_xdata()
    #            yval = trace_filtered.get_ydata()
    #            savetxt('synthetic_data_'+trace_filtered.channel,xval)

            self.add_traces(ntraces)
            self.seis = None

    def __snufflings__():
        return [ ParaEditCp_TF_GTTG() ]


