import sys


def _run_main(app, arguments_list):
    for args in arguments_list:
        argv = sys.argv
        try:
            sys.argv = ['prog']
            app.main(args=args)
        except SystemExit:
            pass
        finally:
            sys.argv = argv


def test_fomosto():
    from pyrocko.apps import fomosto
    import tempfile
    import shutil

    tmp_store = tempfile.mkdtemp(prefix='pyrocko')

    test_arguments = (
        ['--help'],
        ['init', 'qseis.2006b', tmp_store]
        )

    _run_main(fomosto, test_arguments)

    shutil.rmtree(tmp_store)


def test_cake():
    from pyrocko.apps import cake

    test_arguments = (
        ['--help'],
        ['print'],
        ['arrivals'],
        ['paths'],
        ['plot-xt'],
        ['plot-xp'],
        ['plot-rays'],
        ['plot'],
        ['plot-model'],
        ['list-models'],
        ['list-phase-map'],
        ['simplify-model'],
        ['scatter'],
        )

    _run_main(cake, test_arguments)


def test_hamster():
    from pyrocko.apps import hamster

    test_arguments = (
        ['--help'],
        )

    _run_main(hamster, test_arguments)


def test_jackseis():
    from pyrocko.apps import jackseis

    test_arguments = (
        ['--help'],
        )

    _run_main(jackseis, test_arguments)


def test_snuffler():
    from pyrocko.apps import snuffler

    test_arguments = (
        ['--help'],
        )

    _run_main(snuffler, test_arguments)
