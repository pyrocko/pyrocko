import numpy as num
import logging
from datetime import datetime

from pyrocko import gf
from pyrocko import orthodrome as od
from pyrocko.guts import Float, Timestamp, Tuple, StringChoice, Bool, Object

from .base import TargetGenerator
from ..base import Generator, get_gsshg

km = 1e3
d2r = num.pi/180.

logger = logging.getLogger('pyrocko.scenario.base')
guts_prefix = 'pf.scenario'


class SatelliteGeneratorTarget(gf.SatelliteTarget):

    def __init__(self, scene_patch, *args, **kwargs):
        gf.SatelliteTarget.__init__(self, *args, **kwargs)

        self.scene_patch = scene_patch

    def post_process(self, *args, **kwargs):
        resp = gf.SatelliteTarget.post_process(self, *args, **kwargs)

        from kite import Scene
        from kite.scene import SceneConfig, FrameConfig, Meta

        patch = self.scene_patch

        grid, _ = patch.get_grid()

        displacement = num.empty_like(grid)
        displacement.fill(num.nan)
        displacement[patch.get_mask()] = resp.result['displacement.los']

        theta, phi = patch.get_incident_angles()

        llLat, llLon = patch.get_ll_anchor()
        urLat, urLon = patch.get_ur_anchor()
        dLon = num.abs(llLon - urLon) / patch.resolution[0]
        dLat = num.abs(llLat - urLat) / patch.resolution[1]

        scene_config = SceneConfig(
            meta=Meta(
                scene_title='Pyrocko Scenario Generator ({})'
                            .format(datetime.now()),
                orbit_direction=patch.track_direction,
                satellite_name='Sentinel-1'),
            frame=FrameConfig(
                llLon=float(llLon),
                llLat=float(llLat),
                dLat=float(dLat),
                dLon=float(dLon)))

        scene = Scene(
            displacement=displacement,
            theta=theta,
            phi=phi,
            config=scene_config)

        resp.scene = scene

        return resp


class ScenePatch(Object):
    lat_center = Float.T(
        help='Center latitude anchor.')
    lon_center = Float.T(
        help='center longitude anchor.')
    time_master = Timestamp.T(
        help='Timestamp of the master.')
    time_slave = Timestamp.T(
        help='Timestamp of the slave.')
    inclination = Float.T(
        help='Inclination of the satellite orbit towards equatorial plane.')
    apogee = Float.T(
        help='Apogee of the satellite in [m].')
    swath_width = Float.T(
        default=250 * km,
        help='Swath width in [m].')
    track_length = Float.T(
        help='Track length in [m].')
    incident_angle = Float.T(
        help='Near range incident angle in [deg].')
    resolution = Tuple.T(
        help='Resolution of raster in east x north [px].')
    track_direction = StringChoice.T(
        ['Ascending', 'Descending'],
        help='Orbit direction.')
    mask_water = Bool.T(
        default=True,
        help='Mask water bodies.')

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._mask_water = None

    @property
    def width(self):
        track_shift = num.abs(num.cos(self.inclination*d2r)
                              * self.track_length)
        return self.swath_width + track_shift

    def get_ll_anchor(self):
        return od.ne_to_latlon(self.lat_center, self.lon_center,
                               -self.track_length/2, -self.width/2)

    def get_ur_anchor(self):
        return od.ne_to_latlon(self.lat_center, self.lon_center,
                               self.track_length/2, self.width/2)

    def get_ul_anchor(self):
        return od.ne_to_latlon(self.lat_center, self.lon_center,
                               self.track_length/2, -self.width/2)

    def get_lr_anchor(self):
        return od.ne_to_latlon(self.lat_center, self.lon_center,
                               -self.track_length/2, self.width/2)

    def get_corner_coordinates(self):
        inc = self.inclination

        if self.track_direction == 'Ascending':
            llLat, llLon = self.get_ll_anchor()
            urLat, urLon = self.get_ur_anchor()

            ulLat, ulLon = od.ne_to_latlon(
                self.lat_center, self.lon_center,
                self.track_length/2,
                -num.tanh(inc*d2r) * self.width/2)
            lrLat, lrLon = od.ne_to_latlon(
                self.lat_center, self.lon_center,
                -self.track_length/2,
                num.tanh(inc*d2r) * self.width/2)

        elif self.track_direction == 'Descending':
            ulLat, ulLon = self.get_ul_anchor()
            lrLat, lrLon = self.get_lr_anchor()

            urLat, urLon = od.ne_to_latlon(
                self.lat_center, self.lon_center,
                self.track_length/2,
                num.tanh(inc*d2r) * self.width/2)
            llLat, llLon = od.ne_to_latlon(
                self.lat_center, self.lon_center,
                -self.track_length/2,
                -num.tanh(inc*d2r) * self.width/2)

        return ((llLat, llLon), (ulLat, ulLon),
                (urLat, urLon), (lrLat, lrLon))

    def get_grid(self):
        '''Return relative positions of scatterer.

        :param track: Acquisition track, from `'asc'` or `'dsc'`.
        :type track: string
        '''
        easts = num.linspace(0, self.width,
                             self.resolution[0])
        norths = num.linspace(0, self.track_length,
                              self.resolution[1])

        return num.meshgrid(easts, norths)

    def get_mask_track(self):
        east_shifts, north_shifts = self.get_grid()
        norths = north_shifts[:, 0]
        track = num.abs(num.cos(self.inclination*d2r)) * norths

        track_mask = num.logical_and(
            east_shifts > track[:, num.newaxis],
            east_shifts < (track + self.swath_width)[:, num.newaxis])

        if self.track_direction == 'Ascending':
            track_mask = num.fliplr(track_mask)

        return track_mask

    def get_mask_water(self):
        if self._mask_water is None:
            east_shifts, north_shifts = self.get_grid()

            east_shifts -= east_shifts[0, -1]/2
            north_shifts -= north_shifts[-1, -1]/2

            latlon = od.ne_to_latlon(self.lat_center, self.lon_center,
                                     north_shifts.ravel(), east_shifts.ravel())
            points = num.array(latlon).T
            self._mask_water = get_gsshg().get_land_mask(points)\
                .reshape(*east_shifts.shape)

        return self._mask_water

    def get_mask(self):
        mask_track = self.get_mask_track()
        if self.mask_water:
            mask_water = self.get_mask_water()
            return num.logical_and(mask_track, mask_water)
        return mask_track

    def get_incident_angles(self):
        east_shifts, north_shifts = self.get_grid()

        phi = north_shifts.copy()
        phi.fill(self.incident_angle*d2r - num.pi/2)

        east_shifts += num.arctan(self.incident_angle) * self.apogee
        theta = num.tan(self.apogee/east_shifts)

        return theta, phi

    def get_target(self):
        gE, gN = self.get_grid()
        mask = self.get_mask()

        east_shifts = gE[mask].ravel()
        north_shifts = gN[mask].ravel()
        llLat, llLon = self.get_ll_anchor()

        ncoords = east_shifts.size

        theta, phi = self.get_incident_angles()

        theta = theta[mask].ravel()
        phi = phi[mask].ravel()

        if ncoords == 0:
            logger.warning('InSAR taget has no valid points,'
                           ' maybe it\'s all water?')
            return

        return SatelliteGeneratorTarget(
            scene_patch=self,
            lats=num.full(ncoords, fill_value=llLat),
            lons=num.full(ncoords, fill_value=llLon),
            east_shifts=east_shifts,
            north_shifts=north_shifts,
            theta=theta,
            phi=phi)


class AtmosphericNoiseGenerator(Generator):

    amplitude = Float.T(default=1.)

    beta = [5./3, 8./3, 2./3]
    regimes = [.15, .99, 1.]

    def add_athmospheric_noise(self, scene):
        #nE=1024 nN=1024
        #scene = cls()
        scene.meta.title =\
            'Synthetic Displacement | Fractal Noise (Hanssen, 2001)'
        scene = cls._prepareSceneTest(scene, nE, nN)
        if (nE+nN) % 2 != 0:
            raise ArithmeticError('Dimensions of synthetic scene must '
                                  'both be even!')

        dE, dN = (scene.frame.dE, scene.frame.dN)

        rfield = num.random.rand(nE, nN)
        spec = num.fft.fft2(rfield)

        kE = num.fft.fftfreq(nE, dE)
        kN = num.fft.fftfreq(nN, dN)
        k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

        regime = num.array(regime)
        k0 = 0.
        k1 = regime[0] * k_rad.max()
        k2 = regime[1] * k_rad.max()

        r0 = num.logical_and(k_rad > k0, k_rad < k1)
        r1 = num.logical_and(k_rad >= k1, k_rad < k2)
        r2 = k_rad >= k2

        beta = num.array(beta)
        # From Hanssen (2001)
        #   beta+1 is used as beta, since, the power exponent
        #   is defined for a 1D slice of the 2D spectrum:
        #   austin94: "Adler, 1981, shows that the surface profile
        #   created by the intersection of a plane and a
        #   2-D fractal surface is itself fractal with
        #   a fractal dimension  equal to that of the 2D
        #   surface decreased by one."
        beta += 1.
        # From Hanssen (2001)
        #   The power beta/2 is used because the power spectral
        #   density is proportional to the amplitude squared
        #   Here we work with the amplitude, instead of the power
        #   so we should take sqrt( k.^beta) = k.^(beta/2)  RH
        # beta /= 2.

        amp = num.zeros_like(k_rad)
        amp[r0] = k_rad[r0] ** -beta[0]
        amp[r0] /= amp[r0].max()

        amp[r1] = k_rad[r1] ** -beta[1]
        amp[r1] /= amp[r1].max() / amp[r0].min()

        amp[r2] = k_rad[r2] ** -beta[2]
        amp[r2] /= amp[r2].max() / amp[r1].min()

        amp[k_rad == 0.] = amp.max()

        spec *= amplitude * num.sqrt(amp)
        disp = num.abs(num.fft.ifft2(spec))
        disp -= num.mean(disp)

        scene.displacement = disp
        return scene


class InSARDisplacementGenerator(TargetGenerator):
    # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath
    inclination = Float.T(
        default=98.2,
        help='Inclination of the satellite orbit towards equatorial plane'
             ' in [deg]. Defaults to Sentinel-1 (98.1 deg).')
    apogee = Float.T(
        default=693. * km,
        help='Apogee of the satellite in [m]. '
             'Defaults to Sentinel-1 (693 km).')
    swath_width = Float.T(
        default=250 * km,
        help='Swath width in [m]. '
             'Defaults to Sentinel-1 Interfeometric Wide Swath Mode (IW).'
             ' (IW; 250 km).')
    track_length = Float.T(
        default=150 * km,
        help='Track length in [m]. Defaults to 200 km.')
    incident_angle = Float.T(
        default=29.1,
        help='Near range incident angle in [deg]. Defaults to 29.1 deg;'
             ' Sentinel IW mode.')
    resolution = Tuple.T(
        default=(250, 250),
        help='Resolution of raster in east x north [px].')
    mask_water = Bool.T(
        default=True,
        help='Mask out water bodies.')
    noise_generator = Generator.T(
        default=AtmosphericNoiseGenerator.D(),
        help='Add atmospheric noise model after Hansen, 2001.')

    def get_scene_patches(self):
        lat_center, lon_center = self.get_center_latlon()

        scene_patches = []
        for direction in ('Ascending', 'Descending'):
            patch = ScenePatch(
                lat_center=lat_center,
                lon_center=lon_center,
                time_master=0,
                time_slave=0,
                inclination=self.inclination,
                apogee=self.apogee,
                swath_width=self.swath_width,
                track_length=self.track_length,
                incident_angle=self.incident_angle,
                resolution=self.resolution,
                track_direction=direction,
                mask_water=self.mask_water)
            scene_patches.append(patch)

        return scene_patches
