import matplotlib
import matplotlib.pyplot as plt
from pyrocko.gf.seismosizer import BoxcarSTF, TriangularSTF, HalfSinusoidSTF, \
    SmoothRampSTF, ResonatorSTF

font = {'size': 22}
matplotlib.rc('font', **font)


stf_classes = [
    BoxcarSTF, TriangularSTF, HalfSinusoidSTF, SmoothRampSTF, ResonatorSTF]


def plot_stf(stf_cls, duration=5., tref=0., save=True, **kwargs):
    stf_name = stf_cls.__name__

    fig = plt.figure(linewidth=3)
    fig.set_size_inches((9, 5))
    ax = fig.gca()

    stf = stf_cls(duration=duration, **kwargs)
    t, a = stf.discretize_t(deltat=0.05, tref=tref)

    ax.set_title(stf_name)
    ax.plot(t, a, lw=3, color='#004f87')

    ax.axvline(tref, c='k', ls='--', lw=3)

    ax.text(tref + .025 * duration, min(ax.get_ylim()), '$t_{ref}$',
            va='bottom', ha='left', )

    ax.xaxis.grid(alpha=.3, lw=3)
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('Energy')
    ax.set_yticklabels([])

    if save:
        fig.tight_layout()
        fig.savefig('/tmp/stf-%s.svg' % stf_name, format='svg')
    else:
        plt.show()


if __name__ == '__main__':
    for stf in stf_classes:
        plot_stf(stf)
