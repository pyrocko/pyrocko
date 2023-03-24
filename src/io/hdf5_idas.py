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
    'idas_version': 'Version',
    'precice_sampling_freq': 'PreciseSamplingFrequency',
    'receiver_gain': 'ReceiverGain',
    #'continuous_mode': 'Continuous Mode',
    'geo_lat': 'Latitude',
    'geo_lon': 'Longitude',
    'geo_elevation': 'Altitude',

    'unit': 'RawDataUnit' 
}


def get_meta(h5file):
    """
    Get metadata from HDF5 file using (almost) the same fields as for the TDMS files.
    
    Parameters
    ----------
    h5file : HDF5 file object
        The file to extract metadata from
    Returns
    -------
        Dictionary containing the metadata 

    """

    key_list = list(META_KEYS.keys())
    val_list = list(META_KEYS.values())

    meta = dict()
    # the following is based on the PRODML format
    for field in ['Acquisition',
                  'Acquisition/Custom/AdvancedUserSettings',
                  'Acquisition/Custom/SystemSettings',
                  'Acquisition/Custom/SystemInformation/GPS',
                  'Acquisition/Custom/SystemInformation/OSVersion',
                  'Acquisition/Custom/UserSettings',
                  'Acquisition/Raw[0]']:
        try:
            field_keys = h5file[field].attrs.keys()
            for val in val_list:
                if val in field_keys:
                    meta[key_list[val_list.index(val)]] = h5file[field].attrs[val]
        except:
            raise KeyError("Key '%s' not found in PRODML H5 file." % val)
                
    # some type conversions
    for val in ['p_coefficients', 'receiver_gain', 'source_mode', 'unit', 'idas_version']:
        meta[val] = meta[val].decode("utf-8")
    for val in ['decimation_filter', 'normalization']:
        meta[val] = bool(meta[val])
    for val in ['receiver_gain']:
        meta[val] = tuple([float(item) for item in meta[val].split(";")])
    
    return meta


def iload(filename, load_data=True):

    # prevent hard dependency on h5py
    try:
        import h5py
    except Exception as e:
        raise ImportError("Please install 'h5py' to proceed, e.g. by running 'pip install h5py'") from e

    with h5py.File(filename, "r") as f:
        # get the meta data
        meta = get_meta(f)
        # get the actual time series if load_data
        data = f['Acquisition/Raw[0]/RawData'][:] if load_data else None
        # get the sampling rate, starttime, number of samples in space and time
        deltat = 1. / f['Acquisition/Raw[0]'].attrs['OutputDataRate']
        tmin = datetime.fromisoformat(
            f['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'].decode('ascii')).timestamp()
        nchan = f['Acquisition/Raw[0]'].attrs['NumberOfLoci']
        nsamp = f['Acquisition/Raw[0]/RawDataTime'].attrs['Count'] 

    for icha in range(nchan):
        station = '%05i' % icha
        meta_icha = meta.copy()
        meta_icha['channel'] = icha

        tr = trace.Trace(
            network='DA',
            station=station,
            ydata=None,
            deltat=deltat,
            tmin=tmin,
            tmax=tmin + (nsamp - 1) * deltat,
            meta=meta_icha)

        if data is not None:
            tr.set_ydata(data[:,icha])

        yield tr


def detect(first512):
    return first512.startswith(b'\x89HDF')
