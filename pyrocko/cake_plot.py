import math
import numpy as num
from pyrocko import cake

d2r = cake.d2r
r2d = cake.r2d

def globe_cross_section():
    # modified from http://stackoverflow.com/questions/2417794/how-to-make-the-angles-in-a-matplotlib-polar-plot-go-clockwise-with-0-at-the-to

    from matplotlib.projections import PolarAxes, register_projection
    from matplotlib.transforms import Affine2D, Bbox, IdentityTransform

    class GlobeCrossSectionAxes(PolarAxes):
        '''
        A variant of PolarAxes where theta starts pointing north and goes
        clockwise and the radial axis is reversed.
        '''
        name = 'globe_cross_section'

        class GlobeCrossSectionTransform(PolarAxes.PolarTransform):

            def transform(self, tr):
                xy   = num.zeros(tr.shape, num.float_)
                t    = tr[:, 0:1]*d2r
                r    = cake.earthradius - tr[:, 1:2]
                x    = xy[:, 0:1]
                y    = xy[:, 1:2]
                x[:] = r * num.sin(t)
                y[:] = r * num.cos(t)
                return xy

            transform_non_affine = transform

            def inverted(self):
                return GlobeCrossSectionAxes.InvertedGlobeCrossSectionTransform()

        class InvertedGlobeCrossSectionTransform(PolarAxes.InvertedPolarTransform):
            def transform(self, xy):
                x = xy[:, 0:1]
                y = xy[:, 1:]
                r = num.sqrt(x*x + y*y)
                theta = num.arctan2(y, x)*r2d
                return num.concatenate((theta, cake.earthradius-r), 1)

            def inverted(self):
                return GlobeCrossSectionAxes.GlobeCrossSectionTransform()

        def _set_lim_and_transforms(self):
            PolarAxes._set_lim_and_transforms(self)
            self.transProjection = self.GlobeCrossSectionTransform()
            self.transData = (
                self.transScale + 
                self.transProjection + 
                (self.transProjectionAffine + self.transAxes))
            self._xaxis_transform = (
                self.transProjection +
                self.PolarAffine(IdentityTransform(), Bbox.unit()) +
                self.transAxes)
            self._xaxis_text1_transform = (
                self._theta_label1_position +
                self._xaxis_transform)
            self._yaxis_transform = (
                Affine2D().scale(num.pi * 2.0, 1.0) +
                self.transData)
          
            try:
                rlp = getattr(self, '_r_label1_position')
            except AttributeError:
                rlp = getattr(self, '_r_label_position')
            
            self._yaxis_text1_transform = (
                rlp +
                Affine2D().scale(1.0 / 360.0, 1.0) +
                self._yaxis_transform)

    register_projection(GlobeCrossSectionAxes)


tango_colors = {
'butter1': (252, 233,  79),
'butter2': (237, 212,   0),
'butter3': (196, 160,   0),
'chameleon1': (138, 226,  52),
'chameleon2': (115, 210,  22),
'chameleon3': ( 78, 154,   6),
'orange1': (252, 175,  62),
'orange2': (245, 121,   0),
'orange3': (206,  92,   0),
'skyblue1': (114, 159, 207),
'skyblue2': ( 52, 101, 164),
'skyblue3': ( 32,  74, 135),
'plum1': (173, 127, 168),
'plum2': (117,  80, 123),
'plum3': ( 92,  53, 102),
'chocolate1': (233, 185, 110),
'chocolate2': (193, 125,  17),
'chocolate3': (143,  89,   2),
'scarletred1': (239,  41,  41),
'scarletred2': (204,   0,   0),
'scarletred3': (164,   0,   0),
'aluminium1': (238, 238, 236),
'aluminium2': (211, 215, 207),
'aluminium3': (186, 189, 182),
'aluminium4': (136, 138, 133),
'aluminium5': ( 85,  87,  83),
'aluminium6': ( 46,  52,  54)
}

def light(color, factor=0.2):
    return tuple( 1-(1-c)*factor for c in color )

def dark(color, factor=0.5):
    return tuple( c*factor for c in color )

def to01(c):
    return c[0]/255., c[1]/255., c[2]/255.

colors = [ to01(tango_colors[x+i]) for i in '321' for x in 'scarletred chameleon skyblue chocolate orange plum butter'.split() ]
shades = [ light(to01(tango_colors['chocolate1']), i*0.1) for i in xrange(1,9) ]
shades2 = [ light(to01(tango_colors['orange1']), i*0.1) for i in xrange(1,9) ]

def plot_xt(paths, zstart, zstop, plot=None, vred=None, distances=None):
    if distances is not None:
        xmin, xmax = distances.min(), distances.max()
    plot = getplot(plot)
    all_x = []
    all_t = []
    for ipath, path in enumerate(paths):
        if distances is not None:
            if path.xmax() < xmin or path.xmin() > xmax:
                continue
        color = colors[ipath%len(colors)]
        p,x,t = path.draft_pxt(path.endgaps(zstart, zstop))
        if p.size == 0:
            continue
        all_x.append(x)
        all_t.append(t)
        if vred is not None:
            plot.plot(x,t-x/vred, linewidth=2, color=color)
            plot.plot([x[0]], [t[0]-x[0]/vred], 'o', color=color)
            plot.plot([x[-1]], [t[-1]-x[-1]/vred], 'o', color=color)
            plot.text(x[len(x)/2], t[len(x)/2]-x[len(x)/2]/vred, path.used_phase().used_repr(), color=color,
                va='center', ha='center', clip_on=True, bbox=dict(ec=color, fc=light(color), pad=8, lw=1), fontsize=10)
        else:
            plot.plot(x,t, linewidth=2, color=color)
            plot.plot([x[0]], [t[0]], 'o', color=color)
            plot.plot([x[-1]], [t[-1]], 'o', color=color)
            plot.text(x[len(x)/2], t[len(x)/2], path.used_phase().used_repr(), color=color,
                va='center', ha='center', clip_on=True, bbox=dict(ec=color, fc=light(color), pad=8, lw=1), fontsize=10)
   
    all_x = num.concatenate(all_x)
    all_t = num.concatenate(all_t)
    if vred is not None:
        all_t -= all_x/vred
    xxx = num.sort( all_x )
    ttt = num.sort( all_t )
    return xxx.min(), xxx[99*len(xxx)/100], ttt.min(), ttt[99*len(ttt)/100]

def labels_xt(plot=None, vred=None, as_degrees=False):
    plot = getplot(plot)
    if as_degrees:
        plot.xlabel('Distance [deg]')
    else:
        plot.xlabel('Distance [km]')
        xscaled(d2r*cake.earthradius/cake.km, plot)
        
    if vred is None:
        plot.ylabel('Time [s]')
    else:
        if as_degrees:
            plot.ylabel('Time - Distance / %g deg/s [ s ]' % (vred))
        else:
            plot.ylabel('Time - Distance / %g km/s [ s ]' % (d2r*vred*cake.earthradius/cake.km))

def troffset(dx,dy, plot=None):
    plot = getplot(plot)
    from matplotlib import transforms
    return plot.gca().transData + transforms.ScaledTranslation(dx/72., dy/72., plot.gcf().dpi_scale_trans)

def plot_xp(paths, zstart, zstop, plot=None):
    plot = getplot(plot)
    all_x = []
    for ipath, path in enumerate(paths):
        color = colors[ipath%len(colors)]
        p, x, t = path.draft_pxt(path.endgaps(zstart, zstop))
        plot.plot(x, p, linewidth=2, color=color)
        plot.plot(x[:1], p[:1], 'o', color=color)
        plot.plot(x[-1:], p[-1:], 'o', color=color)
        plot.text(x[len(x)/2], p[len(x)/2], path.used_phase().used_repr(), color=color,
                va='center', ha='center', clip_on=True, bbox=dict(ec=color, fc=light(color), pad=8, lw=1))
        all_x.append(x)
    
    xxx = num.sort( num.concatenate(all_x) )
    return xxx.min(), xxx[99*len(xxx)/100] 

def labels_xp(plot=None, as_degrees=False):
    plot = getplot(plot)
    if as_degrees:
        plot.xlabel('Distance [deg]')
    else:
        plot.xlabel('Distance [km]')
        xscaled(d2r*cake.earthradius*0.001, plot)
    plot.ylabel('Ray Parameter [s/deg]')

def labels_model(plot=None):
    plot = getplot(plot)
    plot.xlabel('S-wave and P-wave velocity [km/s]')
    xscaled(0.001, plot)
    plot.ylabel('Depth [km]')
    yscaled(0.001, plot)

def plot_rays(paths, rays, zstart, zstop, plot=None):
    plot = getplot(plot)
    path_to_color = {}
    for ipath, path in enumerate(paths):
        path_to_color[path] = colors[ipath%len(colors)]

    if rays is None:
        rays = paths

    for iray, ray in enumerate(rays):
        if isinstance(ray, cake.RayPath):
            path = ray
            pmin, pmax, xmin, xmax, tmin, tmax = path.ranges(path.endgaps(zstart, zstop))
            if not path._is_headwave:
                p = num.linspace(pmin, pmax, 6)
                x = None
            else:
                x = num.linspace(xmin, xmin*10, 6)
                p = num.atleast_1d(pmin)

            fanz, fanx, _ = path.zxt_path_subdivided(p, path.endgaps(zstart, zstop), x_for_headwave=x)
        else:
            fanz, fanx, _ = ray.zxt_path_subdivided()
            path = ray.path
        
        
        color = path_to_color[path]
        for zs, xs in zip(fanz, fanx):
            l = plot.plot( xs, zs, color=color)


def sketch_model(mod, plot=None):
    from matplotlib import transforms
    plot = getplot(plot)
    ax = plot.gca()
    trans = transforms.BlendedGenericTransform(ax.transAxes, ax.transData)
    
    for dis in mod.discontinuities():
        color = shades[-1]
        plot.axhline( dis.z, color=dark(color), lw=1.5)
        if dis.name is not None:
            plot.text(0.90, dis.z, dis.name, transform=trans, va='center', ha='right', color=dark(color),
                    bbox=dict(ec=dark(color), fc=light(color, 0.3), pad=8, lw=1))

    for ilay, lay in enumerate(mod.layers()):
        if isinstance(lay, cake.GradientLayer):
            tab = shades
        else:
            tab = shades2
        color = tab[ilay%len(tab)]
        plot.axhspan( lay.ztop, lay.zbot, fc=color, ec=dark(color), label='abc')
        if lay.name is not None:
            plot.text(0.95, (lay.ztop + lay.zbot)*0.5, lay.name, transform=trans, va='center', ha='right', color=dark(color),
                    bbox=dict(ec=dark(color), fc=light(color, 0.3), pad=8, lw=1))

def plot_source(zstart, plot=None):
    plot = getplot(plot)
    plot.plot([0], [zstart], 'o', color='black')

def plot_receivers(zstop, distances, plot=None):
    plot = getplot(plot)
    plot.plot(distances, cake.filled(zstop, len(distances)), '^', color='black')

def getplot(plot=None):
    import pylab as lab
    if plot is None:
        return lab
    else:
        return plot

def mk_sc_classes():
    from matplotlib.ticker import FormatStrFormatter, AutoLocator

    class Scaled(FormatStrFormatter):
        def __init__(self, factor):
            FormatStrFormatter.__init__(self, '%g')
            self._factor = factor

        def __call__(self, v, i=0):
            return FormatStrFormatter.__call__(self, v*self._factor, i)

    class ScaledLocator(AutoLocator):
        def __init__(self, factor):
            AutoLocator.__init__(self)
            self._factor = factor

        def bin_boundaries(self, vmin, vmax):
            return [ x/self._factor for x in AutoLocator.bin_boundaries(self, vmin*self._factor, vmax*self._factor) ]

    return Scaled, ScaledLocator
    
def xscaled(factor, plot=None):
    Scaled, ScaledLocator = mk_sc_classes()
    plot = getplot(plot)
    xaxis = plot.gca().xaxis
    xaxis.set_major_formatter( Scaled(factor) )
    xaxis.set_major_locator( ScaledLocator(factor) )

def yscaled(factor, plot=None):
    Scaled, ScaledLocator = mk_sc_classes()
    plot = getplot(plot)
    yaxis = plot.gca().yaxis
    yaxis.set_major_formatter( Scaled(factor) )
    yaxis.set_major_locator( ScaledLocator(factor) )

def labelspace(plot=None):
    plot = getplot(plot)
    xa = plot.gca().get_xaxis()
    ya = plot.gca().get_yaxis()
    for attr in ('labelpad', 'LABELPAD'):
        if hasattr(xa,attr):
            setattr(xa, attr, xa.get_label().get_fontsize())
            setattr(ya, attr, ya.get_label().get_fontsize())
            break

def labels_rays(plot=None, as_degrees=False):
    plot = getplot(plot)
    if as_degrees:
        plot.xlabel('Distance [deg]')
    else:
        plot.xlabel('Distance [km]')
        xscaled(d2r*cake.earthradius/cake.km, plot)
    plot.ylabel('Depth [km]')
    yscaled(1./cake.km, plot)

def plot_surface_efficiency(mat):
    import pylab as lab
    data = []
    for angle in num.linspace(0., 90., 910.):
        pp = math.sin(angle*d2r)/mat.vp
        ps = math.sin(angle*d2r)/mat.vs
        escp = psv_surface(mat, pp, energy=True) 
        escs = psv_surface(mat, ps, energy=True)
        data.append((angle, escp[psv_surface_ind(P,P)], escp[psv_surface_ind(P,S)], 
                            escs[psv_surface_ind(S,S)], escs[psv_surface_ind(S,P)]))

    a,pp,ps,ss,sp = num.array(data).T

    lab.plot(a,pp, label='PP')
    lab.plot(a,ps, label='PS')
    lab.plot(a,ss, label='SS')
    lab.plot(a,sp, label='SP')
    lab.xlabel('Incident Angle')
    lab.ylabel('Energy Normalized Coefficient', position=(-2.,0.5))
    lab.legend()
    lab.show()

def mpl_init():
    import matplotlib
    matplotlib.rcdefaults()
    matplotlib.rc('axes', linewidth=1.5)
    matplotlib.rc('xtick', direction='out')
    matplotlib.rc('ytick', direction='out')
    matplotlib.rc('xtick.major', size=5)
    matplotlib.rc('ytick.major', size=5)
    matplotlib.rc('figure', facecolor='white')

def my_xt_plot(paths, zstart, zstop, distances=None, as_degrees=False, vred=None):
    import pylab as lab
    mpl_init()
    labelspace()
    xmin, xmax, ymin, ymax = plot_xt(paths, zstart, zstop, vred=vred, distances=distances)
    if distances is not None:
        xmin, xmax = distances.min(), distances.max()
    lab.xlim(xmin, xmax)
    lab.ylim(ymin, ymax)
    labels_xt(as_degrees=as_degrees, vred=vred)
    lab.show()

def my_xp_plot(paths, zstart, zstop, distances=None, as_degrees=False):
    import pylab as lab
    mpl_init()
    labelspace()
    xmin, xmax = plot_xp(paths, zstart, zstop) 
    if distances is not None:
        xmin, xmax = distances.min(), distances.max()
    lab.xlim(xmin, xmax)
    labels_xp(as_degrees=as_degrees)
    lab.show()

def my_rays_plot_gcs(mod, paths, rays, zstart, zstop, distances=None):
    import pylab as lab
    mpl_init()
    globe_cross_section()
    plot = lab.subplot(1,1,1, projection='globe_cross_section')
    plot_rays(paths, rays, zstart, zstop, plot=plot)
    plot_source(zstart, plot=plot)
    if distances is not None:
        plot_receivers(zstop, distances, plot=plot)
    lab.ylim(0.,cake.earthradius)
    lab.gca().get_yaxis().set_visible(False)
    lab.show() 

def my_rays_plot(mod, paths, rays, zstart, zstop, distances=None, as_degrees=False):
    import pylab as lab
    mpl_init()
    labelspace()
    plot_rays(paths, rays, zstart, zstop)
    xmin, xmax = lab.xlim()
    ymin, ymax = lab.ylim()
    sketch_model(mod)

    plot_source(zstart)
    if distances is not None:
        plot_receivers(zstop, distances)
    labels_rays(as_degrees=as_degrees)
    mx = (xmax-xmin)*0.05
    my = (ymax-ymin)*0.05
    lab.xlim(xmin-mx, xmax+mx)
    lab.ylim(ymax+my, ymin-my)
    lab.show()

def my_combi_plot(mod, paths, rays, zstart, zstop, distances=None, as_degrees=False, vred=None):
    import pylab as lab 
    from matplotlib.transforms import Affine2D
    mpl_init()
    ax1 = lab.subplot(211)
    labelspace()
    xmin, xmax, ymin, ymax = plot_xt(paths, zstart, zstop, vred=vred, distances=distances)
    if distances is None:
        lab.xlim(xmin, xmax)

    labels_xt(vred=vred, as_degrees=as_degrees)
    lab.setp(ax1.get_xticklabels(), visible=False)
    lab.xlabel('')

    ax2 = lab.subplot(212, sharex=ax1)
    labelspace()
    plot_rays(paths, rays, zstart, zstop)
    xmin, xmax = lab.xlim()
    ymin, ymax = lab.ylim()
    sketch_model(mod)
    
    plot_source(zstart)
    if distances is not None:
        plot_receivers(zstop, distances)
    labels_rays(as_degrees=as_degrees)
    mx = (xmax-xmin)*0.05
    my = (ymax-ymin)*0.05
    ax2.set_xlim(xmin-mx, xmax+mx)
    ax2.set_ylim(ymax+my, ymin-my)
    lab.show()

def my_model_plot(mod):

    import pylab as lab
    mpl_init()
    labels_model()
    sketch_model(mod)
    z = mod.profile('z')
    vp = mod.profile('vp')
    vs = mod.profile('vs')
    lab.plot(vp, z, color=colors[0], lw=2.)
    lab.plot(vs, z, color=colors[2], lw=2.)
    ymin, ymax = lab.ylim()
    xmin, xmax = lab.xlim()
    xmin = 0.
    my = (ymax-ymin)*0.05
    mx = (xmax-xmin)*0.2
    lab.ylim(ymax+my, ymin-my)
    lab.xlim(xmin, xmax+mx)
    lab.show()

