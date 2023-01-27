import h5py
from datetime import datetime
from pyrocko import trace


META_KEYS = {
    'measure_length': 'MeasureLength',
    'start_position': 'StartPosition',
    'spatial_resolution': 'SpatialResolution',
    'fibre_index': 'FibreIndex',
    'fibre_length_multiplier': 'FibreLengthMultiplier',
     #'unit_calibration': 'Unit Calibration (nm)',
    'start_distance': 'StartDistance',
    'stop_distance': 'StopDistance',
    'normalization': 'Normalization',
    'decimation_filter': 'DecimationFilter',
    'gauge_length': 'GaugeLength',
    'norm_offset': 'NormOffset',
    'source_mode': 'SourceMode',
    'time_decimation': 'TimeDecimation',
    'zero_offset': 'ZeroOffset',
    'p_parameter': 'P',
    'p_coefficients': 'P_Coefficients',
    #'idas_version': 'iDASVersion',
    'precice_sampling_freq': 'PreciseSamplingFrequency',
    'receiver_gain': 'ReceiverGain',
    #'continuous_mode': 'Continuous Mode',
    'geo_lat': 'Latitude',
    'geo_lon': 'Longitude',
    'geo_elevation': 'Altitude',

    'channel': None,
    'unit': 'RawDataUnit' 
}


def get_meta(filename):
    """
    Get metadata from HDF5 file using (almost) the same fields as for the TDMS files.
    
    Parameters
    ----------
    filename : str
        Name of the file to extract metadata from
    Returns
    -------
        Dictionary containing the metadata 

    """

    f = h5py.File(filename, "r")
    key_list = list(META_KEYS.keys())
    val_list = list(META_KEYS.values())

    meta = dict()
    # the following is based on the PRODML format
    for field in ['Acquisition',
                  'Acquisition/Custom/AdvancedUserSettings',
                  'Acquisition/Custom/SystemSettings',
                  'Acquisition/Custom/SystemInformation/GPS',
                  'Acquisition/Custom/UserSettings',
                  'Acquisition/Raw[0]']:
        try:
            field_keys = f[field].attrs.keys()
            for val in val_list:
                if val in field_keys:
                    meta[key_list[val_list.index(val)]] = f[field].attrs[val]
        except:
            pass
                
    # some type conversions
    for val in ['p_coefficients', 'receiver_gain', 'source_mode', 'unit']:
        if val in meta.keys():
            meta[val] = meta[val].decode("utf-8")
    for val in ['decimation_filter', 'normalization']:
        if val in meta.keys():
            meta[val] = bool(meta[val])
    
    f.close()
    return meta


def iload(filename, load_data=True):

    meta = get_meta(filename)

    with h5py.File(filename, "r") as f:
        # get the actual time series if load_data
        data = f['Acquisition/Raw[0]/RawData'][:].copy() if load_data else None
        # get the sampling rate, starttime, number of samples in space and time
        deltat = 1. / f['Acquisition/Raw[0]'].attrs['OutputDataRate']
        tmin = datetime.strptime(
            f['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'].decode('ascii'),
            '%Y-%m-%dT%H:%M:%S.%f+00:00').timestamp()
        nchan = f['Acquisition/Raw[0]'].attrs['NumberOfLoci']
        try:
            # usually this should be here?
            nsamp = f['Acquisition/Raw[0]/RawDataTime'].attrs['Count'] 
        except:
            # but sometimes it is here?
            nsamp = f['Acquisition/Raw[0]/RawData'].attrs['Count'] 

    for icha in range(nchan):
        
        assert icha < 99999
        station = '%05i' % icha
        meta['channel'] = icha

        tr = trace.Trace(
            network='DA',
            station=station,
            ydata=None,
            deltat=deltat,
            tmin=tmin,
            tmax=tmin + (nsamp - 1) * deltat,
            meta=meta)

        if data is not None:
            tr.set_ydata(data[:,icha])

        yield tr


def detect(first512):
    return first512.startswith(b'\x89HDF')
