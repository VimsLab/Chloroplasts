import os

from numpy import random
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
import numpy as np
import scipy.spatial as spspat

from overtime.sharedmemory import ndshm
import javabridge
import bioformats
import scipy.signal as spsig
import scipy.stats as spstats
import scipy.misc as spmisc
from overtime.contextual import Parallel, delayed
import itertools
from skimage.color.colorconv import rgb2gray
import warnings


__author__ = 'rhein'

def skeleton(seg):
    skel, dist = skmorph.medial_axis(seg, return_distance=True)
    node, edge, leaf = (spim.label(g, np.ones((3, 3), bool))[0] for g in skel2graph(skel))

    trim_edge = (edge != 0) & ~(skmorph.binary_dilation(node != 0, np.ones((3, 3), bool)) != 0)
    trim_edge = spim.label(trim_edge, np.ones((3, 3), bool))[0]

    leaf_edge_vals = skmorph.binary_dilation(leaf != 0, np.ones((3, 3), bool)) != 0
    leaf_edge_vals = np.unique(trim_edge[leaf_edge_vals])
    leaf_edge_vals = leaf_edge_vals[leaf_edge_vals > 0]
    leaf_edge = leaf != 0

    trim_edge = ndshm.fromndarray(trim_edge)
    leaf_edge = ndshm.fromndarray(leaf_edge)
    Parallel()(
        delayed(set_msk)(leaf_edge, trim_edge, l) for l in leaf_edge_vals)
    trim_edge = np.copy(trim_edge)
    leaf_edge = np.copy(leaf_edge)

    leaf_edge[(skmorph.binary_dilation(leaf_edge, np.ones((3, 3), bool)) != 0) & (edge != 0)] = True
    leaf_edge = spim.label(leaf_edge, np.ones((3, 3), bool))[0]

    leaf_edge_node = skmorph.binary_dilation(leaf_edge != 0, np.ones((3, 3), bool)) != 0
    leaf_edge_node = ((node != 0) & leaf_edge_node) | leaf_edge
    leaf_edge_node = spim.label(leaf_edge_node, np.ones((3, 3), bool))[0]

    cand_node = leaf_edge_node * (node != 0)
    cand_node = cand_node.nonzero()
    cand_node = np.transpose((leaf_edge_node[cand_node],) + cand_node + (2 * dist[cand_node],))

    cand_leaf = leaf_edge_node * (leaf != 0)
    cand_leaf = cand_leaf.nonzero()
    cand_leaf = np.transpose((leaf_edge_node[cand_leaf],) + cand_leaf)

    if len(cand_node) > 0 and len(cand_leaf) > 0:
        cand_leaf = ndshm.fromndarray(cand_leaf)
        cand_node = ndshm.fromndarray(cand_node)
        pruned = Parallel()(
            delayed(prune_leaves)(cand_leaf, cand_node, j) for j in np.unique(cand_node[:, 0]))
        cand_leaf = np.copy(cand_leaf)
        cand_node = np.copy(cand_node)

        pruned_ind = []
        for p in pruned:
            pruned_ind.extend(p)
        pruned_ind = tuple(np.transpose(pruned_ind))

        pruned = ~skel

        pruned = ndshm.fromndarray(pruned)
        leaf_edge = ndshm.fromndarray(leaf_edge)
        Parallel()(
            delayed(set_msk)(pruned, leaf_edge, l) for l in np.unique(leaf_edge[pruned_ind]))
        pruned = np.copy(pruned)
        leaf_edge = np.copy(leaf_edge)

        pruned = ~pruned
    else:
        pruned = skel

    return pruned

def unpad(array, pad_width):
    slc = tuple(slice(start, -stop) for (start, stop) in pad_width)
    return array[slc]


class add_out_arg(object):
    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        out = kwargs.pop('out')
        out[...] = self._function(*args, **kwargs)
        

def hesseig(im, res, sz, nstds, orthstep, nrm=None):
    d2dx2 = get_hessian_kernels(res, sz, nstds, orthstep)
    if nrm is not None:
        d2dx2 = list(d2dx2[i] / np.abs(d2dx2[i]).sum() * nrm[i] for i in range(len(d2dx2)))

    pad = tuple((p, p) for p in np.divide(np.max(list(k.shape for k in d2dx2), 0), 2))
    im_pad = np.pad(im, pad, 'edge').astype(float)

    retval = ndshm.zeros((len(d2dx2),) + im_pad.shape)
    im_pad = ndshm.fromndarray(im_pad)

    Parallel()(
        delayed(add_out_arg(spsig.fftconvolve))(im_pad, k, 'same', out=retval[i])
        for i, k in enumerate(d2dx2))

    retval = np.copy(retval)
    im_pad = np.copy(im_pad)

    d2dx2 = np.empty((len(res),) * 2 + im.shape)
    for ind, (i, j) in enumerate(np.transpose(np.triu_indices(len(res)))):
        d2dx2[i, j] = d2dx2[j, i] = unpad(retval[ind], pad)

    # eigen
    axes = tuple(range(len(res) + 2))
    d2dx2 = np.transpose(d2dx2, axes[2:] + axes[:2])

    retval = ndshm.zeros(d2dx2.shape[:-1])
    d2dx2 = ndshm.fromndarray(d2dx2)

    Parallel()(
        delayed(add_out_arg(np.linalg.eigvalsh))(d2dx2[i], out=retval[i])
        for i in range(len(d2dx2)))

    retval = np.copy(retval)
    d2dx2 = np.concatenate(list(a[None, ...] for a in retval))

    return d2dx2

def differentiation_kernel(order):
    return (-1) ** np.arange(order + 1) * spmisc.comb(order, np.arange(order + 1))


def gaussian_kernel(sigma, num_stds):
    k = np.ones((1,) * len(sigma), float)
    for i, s in enumerate(sigma):
        x = np.ceil(num_stds * s).astype(int)
        x = np.arange(-x, x + 1)
        px = spstats.norm.pdf(x, scale=s)
        px /= px.sum()
        shp = [1] * len(sigma)
        shp[i] = len(px)
        px.shape = tuple(shp)
        k = spsig.convolve(px, k)
    return k


def gaussian_differentiation_kernel(sigma, num_stds, order, delta, scale):
    g = gaussian_kernel(sigma, num_stds)
    d = np.ones((1,) * len(order))
    for i in range(len(order)):
        _d = differentiation_kernel(order[i]).astype(float)
        if len(_d) % 2 != 1:
            _d = (np.pad(_d, ((0, 1),), 'constant') + np.pad(_d, ((1, 0),), 'constant')) / 2.
        _d /= delta[i] ** order[i]
        _d *= np.sqrt(scale[i]) ** order[i]
        shp = np.ones(len(order), int)
        shp[i] = _d.shape[0]
        _d.shape = shp
        d = spsig.convolve(d, _d)
    d = spsig.convolve(g, d)
    return d

def get_hessian_kernels(resolution, size, num_stds, ortho_step_size):
    assert np.all(np.asarray(size) > 1.)
    dd = []
    for dx in np.transpose(np.triu_indices(len(resolution))):
        order = [np.count_nonzero(dx == i) for i in xrange(len(resolution))]
        sigma = np.sum(
            (np.arange(len(resolution), dtype=np.float64) != i) * (ortho_step_size * o)
            for i, o in enumerate(order))
        sigma += (size / 2.) ** 2 - .5 ** 2
        sigma *= (np.min(resolution) / resolution) ** 2
        sigma = np.sqrt(sigma)
        scale = (size / 2.) ** 2 - .5 ** 2
        scale *= (np.min(resolution) / resolution) ** 2
        k = gaussian_differentiation_kernel(sigma, num_stds, order, resolution, scale)
        dd.append(k)
    return dd


def scanal(im, sizes, nstds, orthstep, res):
    nrm = list(np.abs(n).sum() for n in get_hessian_kernels(res, 50., nstds, orthstep))
    scspace = []
    for sz in sizes:
        print sz
        d2dx2 = hesseig(im, res, sz, nstds, orthstep, nrm)
        d2dx2.sort(d2dx2.ndim - 1)
        scspace.append(d2dx2.astype(np.float32))
    scspace = np.concatenate([sc[None, ...] for sc in scspace], 0)
    scspace *= -1
    scspace[scspace < 1e-6] = 0
    scspace = spstats.gmean(scspace, 0)

    lne, crn = scspace[..., 0], scspace[..., 1]
    return lne, crn

def prune_leaves(cand_leaf, cand_node, v):
    l = cand_leaf[:, 0] == v
    l = cand_leaf[l, 1:3]
    n = cand_node[:, 0] == v
    n, s = cand_node[n, 1:3], cand_node[n, 3]
    d = spspat.distance.cdist(l, n)
    d = d.min(1)
    s = s.max()
    return l[d < s]


def set_msk(msk, labels, l):
    msk[labels == l] = True


def skel2graph(skel):
    leaves = [ \
        [[0, 0, 0],
         [0, 1, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[0, 0, 0],
         [0, 1, 1],
         [0, 0, 1]],
        [[0, 0, 0],
         [0, 1, 1],
         [0, 1, 1]]]
    edges = [ \
        [[1, 0, 1],
         [0, 1, 0],
         [0, 0, 0]],
        [[1, 0, 1],
         [0, 1, 1],
         [0, 0, 0]],
        [[1, 0, 1],
         [1, 1, 1],
         [0, 0, 0]],
        [[0, 1, 0],
         [0, 1, 1],
         [0, 0, 0]],
        [[0, 1, 0],
         [0, 1, 1],
         [0, 0, 1]],
        [[1, 1, 0],
         [0, 1, 1],
         [0, 0, 1]],
        [[0, 1, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 1]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 1]],
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, 1, 1],
         [0, 0, 1]],
        [[1, 0, 0],
         [1, 1, 1],
         [0, 0, 1]]]

    def perm2d(k):
        ret = []
        k = np.asarray(k)
        for transp in (k.transpose(0, 1), k.transpose(1, 0)):
            for flip in (transp, transp[:, ::-1], transp[::-1, :], transp[::-1, ::-1]):
                ret.append(flip)
        return ret

    k = 2 ** np.arange(9).reshape((3, 3))
    skel_unq = spim.correlate((skel != 0).astype(int), k.astype(int), mode='constant')

    leaf = np.zeros_like(skel, bool)
    leaf_val = {(l1 * k).sum() for l0 in leaves for l1 in perm2d(l0)}
    for v in leaf_val:
        leaf |= skel_unq == v

    edge = np.zeros_like(skel, bool)
    edge_val = {(e1 * k).sum() for e0 in edges for e1 in perm2d(e0)}
    for v in edge_val:
        edge |= skel_unq == v

    node = (skel != 0) ^ leaf ^ edge

    return node, edge, leaf


def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x


def load_tif_stack(fname, limit=None):
    stack = []
    for i in itertools.count():
        if not limit or i < limit:
            try:
                stack.append(bioformats.load_image(fname, t=i))
            except javabridge.jutil.JavaException:
                break
        else:
            break
    return np.array(stack)

def main():
    stack = load_tif_stack('/home/rhein/workspace/chloroplasts/yfp+2386-tc-1-cytoD_decon_stable_Ch1_yfp.ics movie.tif', limit=None)    
    stack = np.array([rgb2gray(s) for s in stack])
    
    print stack.shape, stack.dtype
    
    for i, im in enumerate(stack):
        lne, _ = scanal(im, .5 + 2 ** np.arange(2, 3), 4., 0., np.ones(2))
#         lne **= .5
        seg = lne > (lne.mean() + 1. * lne.std())
        print i, lne.mean(), lne.std()
        
        skel = skeleton(seg)

#         plt.figure(str(i))
#         plt.subplot(311), plt.imshow(im, 'gray', interpolation='nearest')
#         plt.subplot(312), plt.imshow(skel, 'gray', interpolation='nearest')
#         plt.subplot(313), plt.imshow(seg, 'gray', interpolation='nearest')
        
        seg = (seg != 0).astype(float)
        seg *= 255
        seg = seg.astype(np.uint8)        
        bioformats.write_image('/home/rhein/workspace/chloroplasts/seg.tif', seg, bioformats.PT_UINT8, z=i, size_z=len(stack))

        skel = (skel != 0).astype(float)
        skel *= 255
        skel = skel.astype(np.uint8)        
        bioformats.write_image('/home/rhein/workspace/chloroplasts/skel.tif', skel, bioformats.PT_UINT8, z=i, size_z=len(stack))
        
#         plt.show()
    


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        javabridge.start_vm(args=[], class_path=bioformats.JARS)
        try:
            main()
        finally:
            javabridge.kill_vm()
