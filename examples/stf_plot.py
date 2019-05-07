import matplotlib.pyplot as plt
from pyrocko.gf.seismosizer import BoxcarSTF, TriangularSTF, HalfSinusoidSTF


stf_classes = [BoxcarSTF, TriangularSTF, HalfSinusoidSTF]


def plot_stf(stf_cls, duration=10., save=True):
    fig = plt.figure()
    fig.set_size_inches((8, 4.5))
    ax = fig.gca()

    stf_name = stf_cls.__name__

    tref = 0
    stf = stf_cls(duration=duration, anchor=0.)
    t, a = stf.discretize_t(deltat=0.1, tref=tref)

    ax.set_title(stf_name)
    ax.plot(t, a)

    ax.axvline(*ax.get_ylim(), c='k', ls='--')
    ax.text(tref + .025 * duration, 0., '$t_{ref}$',
            va='center', ha='left', )

    ax.xaxis.grid(alpha=.5)
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('rel. Energy')

    if save:
        fig.savefig('/tmp/stf-%s.svg' % stf_name, format='svg')
    else:
        plt.show()


if __name__ == '__main__':
    for stf in stf_classes:
        plot_stf(stf)
