# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

try:
    import queue
except ImportError:
    import Queue as queue

import multiprocessing
import traceback
import errno


def worker(q_in, q_out, function, eprintignore, pshared):
    kwargs = {}
    if pshared is not None:
        kwargs['pshared'] = pshared

    while True:
        i, args = q_in.get()
        if i is None:
            break

        res, exception = None, None
        try:
            res = function(*args, **kwargs)
        except Exception as e:
            if eprintignore is None or not isinstance(e, eprintignore):
                traceback.print_exc()
            exception = e
        q_out.put((i, res, exception))


def parimap(function, *iterables, **kwargs):
    assert all(
        k in ('nprocs', 'eprintignore', 'pshared') for k in kwargs.keys())

    nprocs = kwargs.get('nprocs', None)
    eprintignore = kwargs.get('eprintignore', 'all')
    pshared = kwargs.get('pshared', None)

    if eprintignore == 'all':
        eprintignore = None

    if nprocs == 1:
        iterables = list(map(iter, iterables))
        kwargs = {}
        if pshared is not None:
            kwargs['pshared'] = pshared

        while True:
            args = []
            for it in iterables:
                try:
                    args.append(next(it))
                except StopIteration:
                    return

            yield function(*args, **kwargs)

        return

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    procs = []

    results = []
    nrun = 0
    nwritten = 0
    iout = 0
    all_written = False
    error_ahead = False
    iterables = list(map(iter, iterables))
    while True:
        if nrun < nprocs and not all_written and not error_ahead:
            args = []
            for it in iterables:
                try:
                    args.append(next(it))
                except StopIteration:
                    pass

            if len(args) == len(iterables):
                if len(procs) < nrun + 1:
                    p = multiprocessing.Process(
                        target=worker,
                        args=(q_in, q_out, function, eprintignore, pshared))
                    p.daemon = True
                    p.start()
                    procs.append(p)

                q_in.put((nwritten, args))
                nwritten += 1
                nrun += 1
            else:
                all_written = True
                [q_in.put((None, None)) for p in procs]
                q_in.close()

        try:
            while nrun > 0:
                if nrun < nprocs and not all_written and not error_ahead:
                    results.append(q_out.get_nowait())
                else:
                    while True:
                        try:
                            results.append(q_out.get())
                            break
                        except IOError as e:
                            if e.errno != errno.EINTR:
                                raise

                nrun -= 1

        except queue.Empty:
            pass

        if results:
            results.sort()
            # check for error ahead to prevent further enqueuing
            if any(exc for (_, _, exc) in results):
                error_ahead = True

            while results:
                (i, r, exc) = results[0]
                if i == iout:
                    results.pop(0)
                    if exc is not None:
                        if not all_written:
                            [q_in.put((None, None)) for p in procs]
                            q_in.close()
                        raise exc
                    else:
                        yield r

                    iout += 1
                else:
                    break

        if all_written and nrun == 0:
            break

    [p.join() for p in procs]
    return
