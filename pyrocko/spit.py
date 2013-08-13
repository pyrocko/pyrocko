import time
import sys
import numpy as num

or_ = num.logical_or
and_ = num.logical_and
not_ = num.logical_not
all_ = num.all

class Cell:
    def __init__(self, tree, index):
        self.tree = tree
        self.tree.ncells += 1
        self.index = index
        self.depths = num.log2(index).astype(num.int)
        self.children = []
        n = 2**self.depths
        i = self.index - n
        delta = (self.tree.xbounds[:,1] - self.tree.xbounds[:,0])/n
        xmin = self.tree.xbounds[:,0]
        self.xbounds = self.tree.xbounds.copy()
        self.xbounds[:,0] = xmin + i * delta
        self.xbounds[:,1] = xmin + (i+1) * delta
        self.a = self.xbounds[:,::-1].copy()
        self.b = self.a.copy()
        self.b[:,1] = self.xbounds[:,1] - self.xbounds[:,0]
        self.b[:,0] = - self.b[:,1]
        it = nditer_outer(tuple(self.xbounds) + (None,))
        for vvv in it:
            vvv[-1][...] = self.tree.f_cached(vvv[:-1])

        self.f = it.operands[-1]
        
    def interpolate(self, x):
        if self.children:
            for cell in self.children:
                if all_(and_(cell.xbounds[:,0] <= x,
                        x <= cell.xbounds[:,1])):
                    return cell.interpolate(x)

        else:
            if all_(num.isfinite(self.f)):
                ws = (x[:,num.newaxis] - self.a)/self.b
                wn = num.multiply.reduce(num.ix_(*ws))
                return num.sum(self.f * wn)
            else:
                return None

    def interpolate_many(self, x):
        if self.children:
            result = num.empty(x.shape[0], dtype=num.float)
            result[:] = None
            for cell in self.children:
                indices = num.where( 3 == num.sum( and_(
                        cell.xbounds[:,0] <= x, x <= cell.xbounds[:,1] ), axis=-1) )[0]

                if indices.size != 0:
                    result[indices] = cell.interpolate_many(x[indices])

            return result

        else:
            if all_(num.isfinite(self.f)):
                ws = (x[...,num.newaxis] - self.a)/self.b
                npoints = ws.shape[0]
                ndim = self.tree.ndim
                ws_pimped = [ ws[:,i,:] for i in range(ndim) ]
                for i in range(ndim):
                    s = [ npoints ] + [ 1 ] * ndim
                    s[1+i] = 2
                    ws_pimped[i].shape = tuple(s)

                wn = ws_pimped[0]
                for idim in range(1,ndim):
                    wn = wn * ws_pimped[idim]

                result = wn * self.f
                for i in range(ndim):
                    result = num.sum(result, axis=-1)

                return result
            else:
                result = num.empty(x.shape[0], dtype=num.float)
                result[:] = None
                return result

    def slice(self, x):
        return [ 
            cell for cell in self.children
                if all_(or_(
                    not_( num.isfinite(x) ),
                    and_(
                        cell.xbounds[:,0] <= x,
                        x <= cell.xbounds[:,1])))
        ]

    def plot_rects(self, plt, x, dims):
        if self.children:
            for cell in self.slice(x):
                cell.plot_rects(plt,x,dims)

        else:
            points = []
            for iy, ix in ((0,0), (0,1), (1,1), (1,0), (0,0)):
                points.append((self.xbounds[dims[0],iy], self.xbounds[dims[1],ix]))

            points = num.transpose(points)
            plt.plot(points[1], points[0], color='black')

    def plot_2d(self, plt, x, dims):

        idims = num.array(dims)
        self.plot_rects(plt, x, dims)
        coords = [ num.linspace(xb[0],xb[1],1+int((xb[1]-xb[0])/d)) for 
                (xb,d) in zip(self.xbounds[idims,:], self.tree.xtols[idims]) ]
        npoints = coords[0].size * coords[1].size
        g = num.meshgrid(*coords[::-1])[::-1]
        points = num.empty((npoints, self.tree.ndim), dtype=num.float)
        for idim in xrange(self.tree.ndim):
            try:
                idimout = dims.index(idim)
                points[:,idim] = g[idimout].ravel()
            except ValueError:
                points[:,idim] = x[idim]

        fi = num.empty((coords[0].size, coords[1].size), dtype=num.float)
        fi_r = fi.ravel()
        fi_r[...] = self.interpolate_many(points)

        if num.any(num.isfinite(fi)):
            fi = num.ma.masked_invalid(fi)
            plt.pcolormesh(coords[1], coords[0], fi, vmin=0.,vmax=1., shading='gouraud')

class SPTree:

    def __init__(self, f, ftol, xbounds, xtols):
        '''Create n-dimensional space partitioning interpolator.
        
        :param f: callable function f(x) where x is a vector of size n
        :param ftol: target accuracy |f_interp(x) - f(x)| <= ftol
        :param xbounds: bounds of x, shape (n,2)
        :param xtols: target coarsenesses in x, vector of size n
        '''

        self.f = f
        self.ftol = float(ftol)
        self.f_values = {}
        self.ncells = 0

        self.xbounds = num.asarray(xbounds, dtype=num.float)
        assert self.xbounds.ndim == 2
        assert self.xbounds.shape[1] == 2
        self.ndim = self.xbounds.shape[0]

        self.xtols = num.asarray(xtols, dtype=num.float)
        assert self.xtols.ndim == 1 and self.xtols.size == self.ndim

        nmaxs = (self.xbounds[:,1] - self.xbounds[:,0]) / self.xtols + 1
        self.maxdepths = num.log2(nmaxs-1).astype(num.int)
        self.root = None
        self.ones_int = num.ones(self.ndim, dtype=num.int)

        cc = num.ix_(*[num.arange(3)]*self.ndim)
        w = num.zeros([3]*self.ndim + [ self.ndim, 2 ])
        for i,c in enumerate(cc):
            w[...,i,0] = (2-c)*0.5
            w[...,i,1] = c*0.5
        
        self.pointmaker = w
        self.pointmaker_mask = num.sum(w[...,0] == 0.5, axis=-1) != 0
        self.pointmaker_masked = w[self.pointmaker_mask]
        
        self.fill()

    def f_cached(self, x):
        return getset(self.f_values, tuple(float(xx) for xx in x), self.f)

    def interpolate(self, x):
        self.root.interpolate(x)

    def interpolate_many(self, x):
        return self.root.interpolate_many(x)

    def fill(self, cell=None):
        if cell is None:
            cell = self.root = Cell(self, self.ones_int)

        
        xtestpoints = num.sum(cell.xbounds * self.pointmaker_masked, axis=-1)
        
        fis = cell.interpolate_many(xtestpoints)
        fes = num.array([ self.f_cached(x) for x in xtestpoints], dtype=num.float)

        works = or_(
                and_(
                    and_(num.isfinite(fes), num.isfinite(fis)),
                    num.abs(fes-fis) < self.ftol ),
                and_(not_(num.isfinite(fes)), not_(num.isfinite(fis))))

        nundef = num.sum(not_(num.isfinite(fes))) + \
                 num.sum(not_(num.isfinite(cell.f)))

        some_undef = 0 < nundef < (xtestpoints.shape[0] + cell.f.size)


        if not all_(works) or some_undef:
            deepen = self.ones_int.copy()
            if True: #not some_undef:
                works_full = num.ones([3]*self.ndim, dtype=num.bool)
                works_full[self.pointmaker_mask] = works
                for idim in range(self.ndim):
                    dimcorners = [ slice(None,None,2) ] * self.ndim 
                    dimcorners[idim] = 1
                    if all_(works_full[dimcorners]):
                        deepen[idim] = 0

            if not num.any(deepen):
                deepen = self.ones_int

            if all_((cell.depths + deepen) < self.maxdepths):
                for iadd in num.ndindex(*(deepen+1)):
                    index_child = (cell.index << deepen) + iadd
                    child = Cell(self, index_child)
                    cell.children.append(child)
                    self.fill(child)

    def plot_2d(self, plt=None, x=None, dims=None):
        assert self.ndim >= 2

        if x is None:
            x = num.zeros(self.ndim)
            x[-2:] = None

        x = num.asarray(x, dtype=num.float)
        if dims is None:
            dims = [ i for (i,v) in enumerate(x) if not num.isfinite(v) ]

        assert len(dims) == 2

        if plt is None:
            from matplotlib import pyplot as p
        else:
            p = plt
            
        self.root.plot_2d(p, x, dims)

        p.xlabel('Dim %i' % dims[1])
        p.ylabel('Dim %i' % dims[0])

        if plt is None:
            p.show()

def getset(d, k, f):
    try: 
        return d[k]
    except KeyError:
        v = d[k] = f(k)
        return v

def nditer_outer(x):
    add = []
    if x[-1] is None:
        x_ = x[:-1]
        add = [ None ]
    else:
        x_ = x

    return num.nditer(x, 
            op_axes=(num.identity(len(x_), dtype=num.int)-1).tolist() + add)


if __name__ == '__main__':

    def f(x):
        x0 = num.array([0.5,0.5,0.5])
        r = 0.6
        if num.sqrt(num.sum((x-x0)**2)) < r:
            
            return  x[2]**4 + x[1]

        return None

    tree = SPTree(f, 0.01, [[0.,1.],[0.,1.],[0.,1.]], [0.01,0.01,0.01])

    from matplotlib import pyplot as plt

    #for i, v in enumerate(num.linspace(0.2, 0.8, 4)):
    #   plt.subplot(2,2,1+i)
    #   tree.plot_2d(plt, x=(v,None,None) )

    v = 0.5
    plt.subplot(2,2,1)
    tree.plot_2d(plt, x=(v,None,None) )
    plt.subplot(2,2,2)
    tree.plot_2d(plt, x=(None,v,None) )
    plt.subplot(2,2,3)
    tree.plot_2d(plt, x=(None,None,v) )

    plt.show()

