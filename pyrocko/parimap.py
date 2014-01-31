import Queue
import itertools, multiprocessing
import traceback

def worker(q_in, q_out, function, eprintignore):
    while True:
        i, args = q_in.get()
        if i is None:
            break

        r, e = None, None
        try:
            r = function(*args)
        except Exception, e:
            if eprintignore is not None and not isinstance(e, eprintignore):
                traceback.print_exc()
            pass 

        q_out.put((i,r,e))

def parimap(function, *iterables, **kwargs):
    assert all( k in ('nprocs', 'eprintignore') for k in kwargs.keys() )

    nprocs = kwargs.get('nprocs', None)
    eprintignore = kwargs.get('eprintignore', 'all')

    if eprintignore == 'all':
        eprintignore = None

    if nprocs == 1:
        for r in itertools.imap(function, *iterables):
            yield r

        return

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=worker, args=(q_in, q_out, function, eprintignore)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    results = []
    nrun = 0
    nwritten = 0
    iout = 0
    all_written = False
    error_ahead = False
    iterables = map(iter, iterables)
    while True:
        if nrun < nprocs and not all_written and not error_ahead:
            args = tuple( it.next() for it in iterables )
            if len(args) == len(iterables):
                q_in.put((nwritten,args))
                nwritten += 1
                nrun += 1
            else:
                all_written = True
                [q_in.put((None,None)) for _ in range(nprocs)]
                [p.join() for p in proc]

        try:
            while nrun > 0:
                if nrun < nprocs and not all_written and not error_ahead:
                    results.append(q_out.get_nowait())
                else:
                    results.append(q_out.get())

                nrun -= 1

        except Queue.Empty:
            pass

        if results:
            results.sort()
            # check for error ahead to prevent further enqueuing
            if any( e for (_,_,e) in results ):
                error_ahead = True

            while results:
                (i,r,e) = results[0]
                if i == iout:
                    results.pop(0)
                    if e: 
                        if not all_written:
                            [q_in.put((None,None)) for _ in range(nprocs)]
                            [p.join() for p in proc]
                        raise e
                    else:
                        yield r

                    iout += 1
                else:
                    break

        if all_written and nrun == 0:
            break

