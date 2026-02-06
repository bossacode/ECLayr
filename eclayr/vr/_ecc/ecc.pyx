cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport ceil
import gudhi as gd
import numpy as np
cimport numpy as cnp
cnp.import_array()


dtype_float = np.float32


cdef class RipsEccBackbone:
    cdef float max_edge_length, resolution, impulse
    cdef Py_ssize_t steps, max_dim

    def __init__(self, float max_edge_length=2, Py_ssize_t max_dim=1, Py_ssize_t steps=32, float impulse=10):
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.steps = steps
        self.resolution = self.max_edge_length / (self.steps - 1)
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
        cdef Py_ssize_t batch_size, d, b, dim, t, vtx_idx_1, vtx_idx_2, i
        cdef float filt
        cdef float[:] vtx_1, vtx_2
        cdef float* grad_1 = NULL
        cdef float* grad_2 = NULL

        batch_size, _, d = x.shape

        cdef float[:, :, :] x_view = x

        ecc = np.zeros((batch_size, self.steps), dtype=dtype_float)
        cdef float[:, :] ecc_view = ecc

        grad_local = np.zeros((*x.shape, self.steps), dtype=dtype_float) if backprop else None    # shape: [B, P, D, steps]
        cdef float[:, :, :, :] grad_local_view = grad_local

        for b in range(batch_size):
            skeleton = gd.RipsComplex(points=x[b], max_edge_length=self.max_edge_length)
            st = skeleton.create_simplex_tree(max_dimension=self.max_dim)
            for vtx_idx, filt in st.get_filtration():
                if filt > self.max_edge_length:
                    break
                dim = len(vtx_idx) - 1  # dimension of simplex
                t = max(<Py_ssize_t>ceil(filt / self.resolution), 0)
                ecc_view[b, t] += (-1.)**dim

                # calculation of gradient only for inputs that require gradient
                if backprop:
                    # vertex
                    if dim == 0:
                        continue
                    # edge
                    elif dim == 1:
                        if grad_1 != NULL:
                            free(grad_1)
                            free(grad_2)

                        vtx_idx_1 = vtx_idx[0]
                        vtx_idx_2 = vtx_idx[1]
                        vtx_1 = x_view[b, vtx_idx_1]    # shape: (d, )
                        vtx_2 = x_view[b, vtx_idx_2]    # shape: (d, )

                        grad_1, grad_2 = RipsEccBackbone._cal_edge_grad(vtx_1, vtx_2, filt, d)

                        for i in range(d):
                            grad_local_view[b, vtx_idx_1, i, t] += grad_1[i] * self.impulse
                            grad_local_view[b, vtx_idx_2, i, t] += grad_2[i] * self.impulse
                    # triangle or higher dimensional simplex
                    else:
                        for i in range(d):
                            grad_local_view[b, vtx_idx_1, i, t] += (-1)**(dim+1) * grad_1[i] * self.impulse
                            grad_local_view[b, vtx_idx_2, i, t] += (-1)**(dim+1) * grad_2[i] * self.impulse
            # free memory after each data
            free(grad_1)
            free(grad_2)
            # cumsum
            for i in range(self.steps - 1):
                ecc_view[b, i+1] += ecc_view[b, i]
        return ecc, grad_local


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