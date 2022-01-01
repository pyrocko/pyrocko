# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function


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
                out.apped('N')


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
    from ..get_terminal_size import get_terminal_size
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
