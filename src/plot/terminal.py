# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from pyrocko import plot


ansi_reverse = u'\033[7m'
ansi_reverse_reset = u'\033[27m'
ansi_dim = u'\033[2m'
ansi_dim_reset = u'\033[22m'

blocks = u'\u2588\u2589\u258a\u258b\u258c\u258d\u258e\u258f '
bar_right = '\u2595'
bar_left = '\u258f'

tri_left = '<'  # '\u25c0'
tri_right = '>'  # '\u25b6'
hmulti = 'X'
vmulti = '%s%%s%s' % (ansi_reverse, ansi_reverse_reset)


def time_axis(view_tmin, view_tmax, sx):

    lines = []
    for napprox in range(10, 0, -1):
        tinc, tunit = plot.nice_time_tick_inc(
            (view_tmax - view_tmin) / napprox)

        times, labels = plot.time_tick_labels(
            view_tmin, view_tmax, tinc, tunit)

        if not labels:
            # show only ticks?
            continue

        delta = (view_tmax - view_tmin) / (sx-2)

        itimes = [int(round((t - view_tmin) / delta)) for t in times]
        nlines = len(labels[0].split('\n'))
        lines = ([bar_right] if itimes[0] != 0 else ['']) * nlines
        overfull = False
        for it, lab in zip(itimes, labels):
            for iline, word in enumerate(lab.split('\n')):
                if len(lines[iline]) > it and word:
                    overfull = True
                    break

                if it > sx-1:
                    break

                lines[iline] += ' ' * (it - len(lines[iline]))
                if len(lines[iline]) < sx-1 and (it - len(lines[iline])) >= 0:
                    lines[iline] += bar_right
                if len(word) > 0 and len(lines[iline]) + len(word) + 1 < sx-1:
                    lines[iline] += word + ' '

            if overfull:
                break

        for iline in range(nlines):
            lines[iline] += ' ' * ((sx-1) - len(lines[iline])) + bar_left

        if overfull:
            continue

        break

    return lines


def bar(view_tmin, view_tmax, changes, tmin, tmax, sx):

    delta = (view_tmax - view_tmin) / (sx-2)
    out = [ansi_dim]
    ic = 0
    while ic < len(changes) and changes[ic][0] < view_tmin:
        ic += 1

    if 0 < ic and ic < len(changes):
        count = changes[ic-1][1]
    else:
        count = 0

    out.append(bar_right if (ic == 0 and view_tmin <= tmin) else tri_left)

    for i in range(sx-2):
        block_tmin = view_tmin + i * delta
        block_tmax = view_tmin + (i+1) * delta
        ic_start = ic
        if i < sx-3:
            while ic < len(changes) and changes[ic][0] < block_tmax:
                ic += 1
        else:
            while ic < len(changes) and changes[ic][0] <= block_tmax:
                ic += 1

        nc_block = ic - ic_start
        if nc_block == 0:
            if count == 0:
                out.append(blocks[-1])
            elif count == 1:
                out.append(blocks[0])
            else:
                out.append(vmulti % ('%i' % count if count <= 9 else 'N'))

        elif nc_block == 1:
            t, new_count = changes[ic_start]
            ib = int(round((t - block_tmin) / delta * (len(blocks) - 1)))
            if new_count == 1 and count == 0:
                out.append(
                    '%s%s%s' % (
                        ansi_reverse, blocks[-ib-1], ansi_reverse_reset))
            elif new_count == 0 and count == 1:
                out.append(blocks[-ib-1])
            elif new_count > count:
                out.append(
                    '%s%s%s' % (
                        ansi_reverse, '+', ansi_reverse_reset))
            elif new_count < count:
                out.append(
                    '%s%s%s' % (
                        ansi_reverse, '-', ansi_reverse_reset))
            elif count == 0 and count == new_count:
                out.append(blocks[-1])

            elif count == 1 and count == new_count:
                out.append(blocks[0])
            else:
                out.append('N')

            count = new_count

        elif nc_block > 1:
            _, count = changes[ic_start + nc_block - 1]
            out.append(hmulti)
        else:
            assert False

    out.append(
        bar_left if (ic == len(changes) and tmax <= view_tmax) else tri_right)
    out.append(ansi_dim_reset)

    return ''.join(out)


if __name__ == '__main__':
    from shutil import get_terminal_size
    sx, _ = get_terminal_size()

    view_tmin = 0.
    view_tmax = 100.

    import numpy as num

    n = 20
    phis = num.linspace(-0.1, 1., n)

    for phi in phis:
        tmin = view_tmin + 0.2 * phi * (view_tmax - view_tmin)
        tmin2 = view_tmin + 0.3 * phi * (view_tmax - view_tmin)
        tmax2 = view_tmax - 0.3 * phi * (view_tmax - view_tmin)
        tmax = view_tmax - 0.2 * phi * (view_tmax - view_tmin)
        print(bar(
            view_tmin, view_tmax,
            [[tmin, 1], [tmin2, 2], [tmax2, 1], [tmax, 0]],
            view_tmin, view_tmax, sx))

    import time
    import numpy as num
    t0 = time.time()
    exps = num.linspace(-4., 11., 100)

    for exp in exps:

        for line in time_axis(t0, t0+10**exp, sx):
            print(line)
        print()

    from pyrocko import util
    tmin = util.stt('2011-04-15 10:05:14')
    tmax = util.stt('2012-09-20 14:41:00')
    for line in time_axis(tmin, tmax, sx):
        print(line)
    print()

    tmin = util.stt('2011-11-27 00:00:00')
    tmax = util.stt('2011-11-28 00:00:00')
    for line in time_axis(tmin, tmax, sx):
        print(line)
    print()
