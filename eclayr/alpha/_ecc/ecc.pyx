cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport ceil
import gudhi as gd
import numpy as np
cimport numpy as cnp
cnp.import_array()


dtype_float = np.float32


cdef class AlphaEccBackbone:
    cdef float t_min, t_max, resolution, impulse
    cdef Py_ssize_t steps

    def __init__(self, interval=[0, 1], Py_ssize_t steps=32, float impulse=10):
        self.t_min, self.t_max = interval
        self.steps = steps
        self.resolution = (self.t_max - self.t_min) / (self.steps - 1)
        self.impulse = impulse

    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    def cal_ecc(self, object x, bint backprop):
        """_summary_

        Args:
            x (numpy.ndarray): Point cloud of shape [B, P, D]
            backprop (bool): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        cdef Py_ssize_t batch_size, b, dim, t, i
        cdef float filt

        batch_size, _, d = x.shape

        cdef float[:, :, :] x_view = x

        ecc = np.zeros((batch_size, self.steps), dtype=dtype_float)
        cdef float[:, :] ecc_view = ecc

        grad_local = np.zeros((*x.shape, self.steps), dtype=dtype_float) if backprop else None    # shape: [B, P, D, steps]
        cdef float[:, :, :, :] grad_local_view = grad_local

        for b in range(batch_size):
            skeleton = gd.AlphaComplex(points=x[b])
            st = skeleton.create_simplex_tree()
            # for vtx_idx, filt in reversed(list(st.get_filtration())):  # decreasing order
            for vtx_idx, filt in st.get_filtration():
                if filt > self.t_max:
                    break
                dim = len(vtx_idx) - 1  # dimension of simplex
                t = max(<Py_ssize_t>ceil((filt - self.t_min) / self.resolution), 0)
                ecc_view[b, t] += (-1.)**dim

                # calculation of gradient only for inputs that require gradient
                # if backprop:
                #     # vertex
                #     if dim == 0:
                #         continue
                #     # edge
                #     elif dim == 1:
                #         if filt == prev_filt:   # attached by higher dimensional simplex
                #             ind = [i in vtx_idx for i in prev_vtx_idx]
                #             grad = prev_grad[ind]
                #             assert len(grad) != 0, "Violation of Alpha general position assumption"
                #             grad_local[vtx_idx, :, t] -= grad
                #         else:                   # attaching simplex
                #             vtx_1, vtx_2 = x[vtx_idx]
                #             grad = np.stack([(vtx_1 - vtx_2)/2, (vtx_2 - vtx_1)/2], axis=0)
                #             grad_local[vtx_idx, :, t] -= grad
                #     # triangle
                #     elif dim == 2:
                #         vtx_1, vtx_2, vtx_3 = x[vtx_idx]
                #         grad = np.stack([self._grad_u(vtx_1, vtx_2, vtx_3), self._grad_v(vtx_1, vtx_2, vtx_3), self._grad_w(vtx_1, vtx_2, vtx_3)], axis=0)
                #         grad_local[vtx_idx, :, t] += grad
                #         prev_filt = filt
                #         prev_vtx_idx = vtx_idx
                #         prev_grad = grad
                #     # tetrahedron or higher dimensional simplex
                #     else:
                #         raise NotImplementedError("Backpropagation not implemented for 3-simplex and higher")
            # cumsum
            for i in range(self.steps - 1):
                ecc_view[b, i+1] += ecc_view[b, i]
        return ecc, grad_local



    @staticmethod
    def _grad_u(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. u.

        Args:
            u (_type_, optional): _description_.
            v (_type_, optional): _description_.
            w (_type_, optional): _description_.

        Returns:
            _type_: _description_
        """
        t_0 = (u - v)
        t_1 = (np.linalg.norm(t_0) ** 2)
        t_2 = np.linalg.norm((v - w))
        t_3 = (t_2 ** 2)
        t_4 = (u - w)
        t_5 = (np.linalg.norm(t_4) ** 2)
        t_6 = (4 * t_1)
        t_7 = ((t_1 + t_3) - t_5)
        t_8 = ((t_6 * t_3) - (t_7 ** 2))
        t_9 = (t_8 ** 2)
        gradient = ((((((2 * t_3) * t_5) / t_8) * t_0) + ((((2 * t_1) * t_3) / t_8) * t_4)) - (((((8 * t_1) * ((t_2 ** 4) * t_5)) / t_9) * t_0) - ((((t_6 * ((t_3 * t_5) * t_7)) / t_9) * t_0) - (((4 * ((t_5 * t_1) * (t_3 * t_7))) / t_9) * t_4))))
        return gradient

    @staticmethod
    def _grad_v(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. v.

        Args:
            u (_type_, optional): _description_. Defaults to u.
            v (_type_, optional): _description_. Defaults to v.
            w (_type_, optional): _description_. Defaults to w.

        Returns:
            _type_: _description_
        """
        t_0 = (u - v)
        t_1 = np.linalg.norm(t_0)
        t_2 = (t_1 ** 2)
        t_3 = (v - w)
        t_4 = np.linalg.norm(t_3)
        t_5 = (t_4 ** 2)
        t_6 = (np.linalg.norm((u - w)) ** 2)
        t_7 = ((t_2 + t_5) - t_6)
        t_8 = (((4 * t_2) * t_5) - (t_7 ** 2))
        t_9 = (t_8 ** 2)
        t_10 = (t_2 * t_5)
        gradient = ((((((2 * t_2) * t_6) / t_8) * t_3) - (((2 * (t_5 * t_6)) / t_8) * t_0)) - ((((((8 * t_5) * ((t_1 ** 4) * t_6)) / t_9) * t_3) - (((8 * ((t_2 * (t_4 ** 4)) * t_6)) / t_9) * t_0)) - (((((4 * t_5) * ((t_2 * t_6) * t_7)) / t_9) * t_3) - (((4 * (t_10 * (t_6 * t_7))) / t_9) * t_0))))
        return gradient

    @staticmethod
    def _grad_w(u, v, w):
        """Given 3 points  that form a triangle in Alpha Complex, compute the gradient w.r.t. w.

        Args:
            u (_type_): _description_
            v (_type_): _description_
            w (_type_): _description_

        Returns:
            _type_: _description_
        """
        t_0 = np.linalg.norm((u - v))
        t_1 = (t_0 ** 2)
        t_2 = (v - w)
        t_3 = (np.linalg.norm(t_2) ** 2)
        t_4 = (u - w)
        t_5 = (np.linalg.norm(t_4) ** 2)
        t_6 = ((t_1 + t_3) - t_5)
        t_7 = (((4 * t_1) * t_3) - (t_6 ** 2))
        t_8 = (2 * t_1)
        t_9 = (t_1 * t_3)
        t_10 = (t_7 ** 2)
        gradient = -(((((t_8 * t_5) / t_7) * t_2) + (((t_8 * t_3) / t_7) * t_4)) - ((((((8 * t_3) * ((t_0 ** 4) * t_5)) / t_10) * t_2) + ((((4 * t_5) * (t_9 * t_6)) / t_10) * t_4)) - (((4 * ((t_3 * t_1) * (t_5 * t_6))) / t_10) * t_2)))
        return gradient

    @staticmethod
    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    cdef inline (float*, float*) _cal_edge_grad(float[:] vtx_1, float[:] vtx_2, float filt, int d):
        cdef float *grad_1 = <float *> malloc(d * sizeof(float))
        cdef float *grad_2 = <float *> malloc(d * sizeof(float))

        for i in range(d):
            grad_1[i] = (vtx_1[i] - vtx_2[i]) / filt
            grad_2[i] = (vtx_2[i] - vtx_1[i]) / filt

        return grad_1, grad_2