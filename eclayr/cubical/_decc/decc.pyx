cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport ceil, exp
import gudhi as gd
import numpy as np
cimport numpy as cnp
cnp.import_array()


dtype_float = np.float32


@cython.cdivision(True)     # turn off checking for division by zero
cdef inline float sigmoid(float x, float lam) except -1:
    return 1 / (1 + exp(-lam * x))


cdef class DECC:
    cdef Py_ssize_t H, W, cH, cW, num_cells, steps
    cdef float t_min, t_max, res, lb, ub, lam
    cdef float[:] tseq
    cdef const float[:] cell_dims

    def __init__(self, arr_size, interval=[0., 1.], steps=32, lam=200.):
        """
        Args:
            arr_size (Iterable[int], optional): (H, W) corresponding to the height and width of input array.
            interval (Iterable[float], optional): Interval of filtration values to be considered. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            lam (float, optional): Controls the tightness of sigmoid approximation. Defaults to 200.
        """
        self.H, self.W = arr_size
        self.cH, self.cW = [2*i-1 for i in arr_size]    # size of the cubical complex
        self.num_cells = self.cH * self.cW
        self.t_min, self.t_max = interval
        self.steps = steps
        self.res = (self.t_max-self.t_min) / (steps-1)  # distance between adjacent grid points in interval
        self.lb = self.t_min - self.res         # lower bound for not skipping gradient computation during backpropagation
        self.ub = self.t_max + self.res         # upper bound for not skipping gradient computation during backpropagation
        self.lam = lam
        self.tseq = np.linspace(*interval, steps, dtype=dtype_float)
        self.cell_dims = self._set_cell_dims()  # dimension of each cell in the cubical complex

    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    @cython.initializedcheck(False) # turn off initialization check
    def cal_ecc(self, float[:, :, :, :] x, bint backprop):
        """Compute the sigmoid-approximated Euler characteristic curve of 2D images using cubical complex.

        Args:
            x (ndarray or tensor of shape (B, C, H, W)): Batch of 2D input images.
            backprop (bint): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        cdef Py_ssize_t num_channels = x.shape[1]
        cdef Py_ssize_t b, c, cell, i, j, pix, num_max
        cdef float filt, dim, coef, sig
        cdef Py_ssize_t *neighbor_vtx
        cdef Py_ssize_t *vtx
        cdef float[4] vtx_filt
        
        ecc = np.zeros((batch_size, num_channels, self.steps), dtype=dtype_float)
        cdef float[:, :, :] ecc_view = ecc

        grad = np.zeros((batch_size, num_channels, self.H*self.W, self.steps), dtype=dtype_float) if backprop else None
        cdef float[:, :, :, :] grad_view = grad

        cdef float[:] filtration

        for b in range(batch_size):         # iterate over batch
            for c in range(num_channels):   # iterate over channel
                cub_cpx = gd.CubicalComplex(vertices=x[b, c])   # V-contruction
                filtration = cub_cpx.all_cells().astype(dtype_float).flatten()
                for cell in range(self.num_cells):  # iterate over all cells in cubical complex
                    filt = filtration[cell]
                    if filt > self.ub:      # doesn't affect neither forward nor backward path
                        continue
                    dim = self.cell_dims[cell]
                    coef = (-1.)**dim

                    # calculate ecc using sigmoid approximation
                    for i in range(self.steps):
                        ecc_view[b, c, i] += coef * sigmoid(self.tseq[i]-filt, self.lam)
                    
                    # compute gradient only for inputs that require backpropagation
                    if backprop:
                        if filt < self.lb:  # doesn't affect backward path
                            continue
                        # vertex
                        if dim == 0:
                            pix = self._vtx2pix(cell)                           # index of the corresponding pixel in flattened original image
                            for i in range(self.steps):
                                sig = sigmoid(self.tseq[i]-filt, self.lam)
                                grad_view[b, c, pix, i] -= self.lam * sig * (1-sig)
                        # edge
                        elif dim == 1:
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of 2 neighbor vertices
                            for i in range(2):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            if vtx_filt[0] == vtx_filt[1]:  # split gradient when the neighboring vertices have the same filtration value
                                for i in range(2):
                                    pix = self._vtx2pix(neighbor_vtx[i])
                                    for j in range(self.steps):
                                        sig = sigmoid(self.tseq[j]-filt, self.lam)
                                        grad_view[b, c, pix, j] += self.lam * sig * (1-sig) / 2.
                            else:
                                i = 0 if vtx_filt[0] > vtx_filt[1] else 1
                                pix = self._vtx2pix(neighbor_vtx[i])
                                for j in range(self.steps):
                                    sig = sigmoid(self.tseq[j]-filt, self.lam)
                                    grad_view[b, c, pix, j] += self.lam * sig * (1-sig)
                            free(neighbor_vtx)
                        # square
                        else:
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of 4 neighbor vertices
                            for i in range(4):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            vtx = self._find_max_vtx(vtx_filt, neighbor_vtx, 4, &num_max)   # vtx points at a C array containing index of vertices that contribute to constructing the cell
                            for i in range(num_max):
                                pix = self._vtx2pix(vtx[i])
                                for j in range(self.steps):
                                    sig = sigmoid(self.tseq[j]-filt, self.lam)
                                    grad_view[b, c, pix, j] -= self.lam * sig * (1-sig) / num_max
                            free(vtx)
                            free(neighbor_vtx)
        return ecc, grad

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t _vtx2pix(self, Py_ssize_t vtx):
        """Given the index of a vertex, this function returns the index of the corresponding pixel.

        Args:
            vtx (Py_ssize_t): Index of vertex.

        Returns:
            Py_ssize_t: Index of corresponding pixel.
        """
        return (vtx // (2*self.cW))*self.W + (vtx % self.cW)/2

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t* _find_neighbor_vtx(self, Py_ssize_t cell, float dim):
        """Returns the indices of a cell's neighboring vertices. Not used for cells that are already vertices.

        Args:
            cell (Py_ssize_t): Index of cell.
            dim (float): Dimension of cell.

        Returns:
            neighbor_vtx (Pointer[Py_ssize_t]): C array containing index of neighboring edges or squares.
        """
        cdef Py_ssize_t row_num
        cdef Py_ssize_t *neighbor_vtx = <Py_ssize_t *> malloc(<Py_ssize_t>dim * 2 * sizeof(Py_ssize_t)) # assign size 2 array for edges and size 4 array for squares
        # edge
        if dim == 1:
            row_num = cell // self.cW
            if row_num % 2 == 0:    # even row
                neighbor_vtx[:] = [cell-1, cell+1]
            else:                   # odd row
                neighbor_vtx[:] = [cell-self.cW, cell+self.cW]
        # square
        else:
            neighbor_vtx[:] = [cell-self.cW-1, cell-self.cW+1, cell+self.cW-1, cell+self.cW+1]
        return neighbor_vtx

    cdef inline Py_ssize_t* _find_max_vtx(self, float *vtx_filt, Py_ssize_t *neighbor_vtx, Py_ssize_t arr_size, Py_ssize_t *num_max):
        """Returns the index (indices) of a cell's maximum neighboring vertex (vertices). Not used for cells that are already vertices.

        Args:
            vtx_filt (float): Filtration value of neighbor vertices.
            neighbor_vtx (Pointer[Py_ssize_t]): C array containing index of neighbor vertices.
            size (Py_ssize_t): Number of neighboring vertices.
            num_max 

        Returns:
            vtx (Pointer[Py_ssize_t]): C array containing index of vertices that contribute to constructing the cell.

        """
        cdef float max_val = vtx_filt[0]
        cdef Py_ssize_t j = 0, count = 0

        # find maximum filtration value
        for i in range(1, arr_size):
            if vtx_filt[i] > max_val:
                max_val = vtx_filt[i]
        
        # count how many times max_val occurs
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                count += 1

        cdef Py_ssize_t *vtx = <Py_ssize_t *> malloc(count * sizeof(Py_ssize_t))
        
        # store the index of vertices that have maximum filtration value
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                vtx[j] = neighbor_vtx[i]
                j += 1

        # number of max_val occurences
        num_max[0] = count
        return vtx

    cdef _set_cell_dims(self):
        """Sets dimension for all cells in the cubical complex. Dimensions of vertices, edges, and squares are 0, 1, and 2, respectively.
        Even rows consist of (vertex, edge, vertex, ..., edge, vertex) and odd rows consist of (edge, square, edge, ..., square, edge).

        Returns:
            cell_dims (ndarray of shape (cub_h*cub_w, )): Dimension of all cells in the cubical complex.
        """
        cell_dims = np.zeros([self.cH, self.cW], dtype=dtype_float)
        cell_dims[[i for i in range(self.cH) if i % 2 == 1], :] += 1
        cell_dims[:, [i for i in range(self.cW) if i % 2 == 1]] += 1
        cell_dims.setflags(write=False)
        return cell_dims.flatten()


cdef class DECC3d:
    cdef Py_ssize_t D, H, W, cD, cH, cW, num_cells, num_cells_D, num_pix_D, steps
    cdef float t_min, t_max, res, lb, ub, lam
    cdef float[:] tseq
    cdef const float[:] cell_dims

    def __init__(self, arr_size, interval=[0., 1.], steps=32, constr="V", lam=200.):
        """
        Args:
            arr_size (Iterable[int], optional): [H, W] of input array.
            interval (list[float], optional): Minimum and maximum value of interval to consider. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            constr (str, optional): One of V or T, corresponding to V-construction and T-construction, respectively. Defaults to V.
            lam (float, optional): Controls the tightness of sigmoid approximation. Defaults to 200.
        """
        self.D, self.H, self.W = arr_size
        self.cD, self.cH, self.cW = [2*i-1 for i in arr_size]   # size of the cubical complex
        self.num_cells = self.cD * self.cH * self.cW
        self.num_cells_D = self.cH * self.cW    # number of cells for each depth
        self.num_pix_D = self.H * self.W        # number of pixels for each depth
        self.t_min, self.t_max = interval
        self.steps = steps
        self.res = (self.t_max-self.t_min) / (steps-1)  # distance between adjacent grid points in interval
        self.lb = self.t_min - self.res         # lower bound for not skipping gradient computation during backpropagation
        self.ub = self.t_max + self.res         # upper bound for not skipping gradient computation during backpropagation
        self.lam = lam
        self.tseq = np.linspace(*interval, steps, dtype=dtype_float)
        self.cell_dims = self._set_cell_dims()  # dimension of each cell in the cubical complex

    # V-construction
    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    @cython.initializedcheck(False) # turn off initialization check
    def cal_ecc(self, float[:, :, :, :, :] x, bint backprop):
        """Compute the sigmoid-approximated Euler characteristic curve of 2D images using cubical complex.

        Args:
            x (ndarray or tensor of shape (B, C, D, H, W)): Batch of 3D input images.
            backprop (bint): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        cdef Py_ssize_t num_channels = x.shape[1]
        cdef Py_ssize_t b, c, cell, i, j, pix, n_neighbor_vtx, num_max
        cdef float filt, dim, coef, sig
        cdef Py_ssize_t *neighbor_vtx
        cdef Py_ssize_t *vtx
        cdef float[8] vtx_filt
        
        ecc = np.zeros((batch_size, num_channels, self.steps), dtype=dtype_float)
        cdef float[:, :, :] ecc_view = ecc

        grad = np.zeros((batch_size, num_channels, self.D*self.H*self.W, self.steps), dtype=dtype_float) if backprop else None
        cdef float[:, :, :, :] grad_view = grad

        cdef float[:] filtration

        for b in range(batch_size):         # iterate over batch
            for c in range(num_channels):   # iterate over channel
                cub_cpx = gd.CubicalComplex(vertices=x[b, c])   # V-contruction
                filtration = cub_cpx.all_cells().astype(dtype_float).flatten()
                for cell in range(self.num_cells):  # iterate over all cells in cubical complex
                    filt = filtration[cell]
                    if filt > self.ub:      # doesn't affect neither forward nor backward path
                        continue 
                    dim = self.cell_dims[cell]
                    coef = (-1.)**dim

                    # calculate ecc using sigmoid approximation
                    for i in range(self.steps):
                        ecc_view[b, c, i] += coef * sigmoid(self.tseq[i]-filt, self.lam)

                    # calculation of gradient only for inputs that require gradient
                    if backprop:
                        if filt < self.lb:  # doesn't affect backward path
                            continue
                        # vertex
                        if dim == 0:
                            pix = self._vtx2pix(cell)                           # index of the corresponding pixel in flattened original image
                            for i in range(self.steps):
                                sig = sigmoid(self.tseq[i]-filt, self.lam)
                                grad_view[b, c, pix, i] -= self.lam * sig * (1 - sig)
                        # edge
                        elif dim == 1:
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of 2 neighbor vertices
                            for i in range(2):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            if vtx_filt[0] == vtx_filt[1]:  # split gradient when the neighboring vertices have the same filtration value
                                for i in range(2):
                                    pix = self._vtx2pix(neighbor_vtx[i])
                                    for j in range(self.steps):
                                        sig = sigmoid(self.tseq[j]-filt, self.lam)
                                        grad_view[b, c, pix, j] += self.lam * sig * (1-sig) / 2.
                            else:
                                i = 0 if vtx_filt[0] > vtx_filt[1] else 1
                                pix = self._vtx2pix(neighbor_vtx[i])
                                for j in range(self.steps):
                                    sig = sigmoid(self.tseq[j]-filt, self.lam)
                                    grad_view[b, c, pix, j] += self.lam * sig * (1-sig)
                            free(neighbor_vtx)
                        # square and cube
                        else:
                            n_neighbor_vtx = 4 if dim == 2 else 8               # 4 neighbor vertices if square, 8 neighbor vertices if cube
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of "n_neighbor_vtx" neighbor vertices
                            for i in range(n_neighbor_vtx):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            vtx = self._find_max_vtx(vtx_filt, neighbor_vtx, n_neighbor_vtx, &num_max)   # vtx points at a C array containing index of vertices that contribute to constructing the cell
                            for i in range(num_max):
                                pix = self._vtx2pix(vtx[i])
                                if dim == 2:    # square
                                    for j in range(self.steps):
                                        sig = sigmoid(self.tseq[j]-filt, self.lam)
                                        grad_view[b, c, pix, j] -= self.lam * sig * (1-sig) / num_max
                                else:           # cube
                                    for j in range(self.steps):
                                        sig = sigmoid(self.tseq[j]-filt, self.lam)
                                        grad_view[b, c, pix, j] += self.lam * sig * (1-sig) / num_max
                            free(vtx)
                            free(neighbor_vtx)
        return ecc, grad

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t _vtx2pix(self, Py_ssize_t vtx):
        """Given the index of a vertex, this function returns the index of the corresponding pixel.

        Args:
            vtx (Py_ssize_t): Index of vertex.

        Returns:
            Py_ssize_t: Index of corresponding pixel.
        """
        cdef Py_ssize_t leftover = vtx % (2*self.num_cells_D)
        return (vtx // (2*self.num_cells_D))*(self.num_pix_D) + (leftover // (2*self.cW))*self.W + (leftover % self.cW)/2

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t* _find_neighbor_vtx(self, Py_ssize_t cell, float dim):
        """Returns the indices of a cell's neighboring vertices. Not used for cells that are already vertices.

        Args:
            cell (Py_ssize_t): Index of cell.
            dim (float): Dimension of cell.

        Returns:
            neighbor_vtx (Pointer[Py_ssize_t]): C array containing index of neighboring edges, squares, or cubes.
        """
        cdef Py_ssize_t depth_num, row_num, leftover
        cdef Py_ssize_t *neighbor_vtx = <Py_ssize_t *> malloc(<Py_ssize_t>(2 ** dim) * sizeof(Py_ssize_t)) # assign size 2 array for edges, size 4 array for squares, and size 8 for cubes
        # cube
        if dim ==3:
            neighbor_vtx[:] = [cell-self.num_cells_D-self.cW-1, cell-self.num_cells_D-self.cW+1, cell-self.num_cells_D+self.cW-1, cell-self.num_cells_D+self.cW+1,
                            cell+self.num_cells_D-self.cW-1, cell+self.num_cells_D-self.cW+1, cell+self.num_cells_D+self.cW-1, cell+self.num_cells_D+self.cW+1]
        else:
            depth_num = cell // self.num_cells_D
            leftover = cell % self.num_cells_D
            if depth_num % 2 == 0:  # even depth
                # edge
                if dim == 1:
                    row_num = leftover // self.cW
                    if row_num % 2 == 0:    # even row
                        neighbor_vtx[:] = [cell-1, cell+1]
                    else:                   # odd row
                        neighbor_vtx[:] = [cell-self.cW, cell+self.cW]
                # square
                else:
                    neighbor_vtx[:] = [cell-self.cW-1, cell-self.cW+1, cell+self.cW-1, cell+self.cW+1]
            else:                   # odd depth
                # edge
                if dim == 1:
                    neighbor_vtx[:] = [cell-self.num_cells_D, cell+self.num_cells_D]
                # square
                else:
                    row_num = leftover // self.cW
                    if row_num % 2 == 0:    # even row
                        neighbor_vtx[:] = [cell-self.num_cells_D-1, cell-self.num_cells_D+1, cell+self.num_cells_D-1, cell+self.num_cells_D+1]
                    else:
                        neighbor_vtx[:] = [cell-self.num_cells_D-self.cW, cell-self.num_cells_D+self.cW, cell+self.num_cells_D-self.cW, cell+self.num_cells_D+self.cW]
        return neighbor_vtx

    cdef inline Py_ssize_t* _find_max_vtx(self, float *vtx_filt, Py_ssize_t *neighbor_vtx, Py_ssize_t arr_size, Py_ssize_t *num_max):
        """Returns the index (indices) of a cell's maximum neighboring vertex (vertices). Not used for cells that are already vertices.

        Args:
            vtx_filt (float): Filtration value of neighbor vertices.
            neighbor_vtx (Pointer[Py_ssize_t]): C array containing index of neighbor vertices.
            size (Py_ssize_t): Number of neighboring vertices.
            num_max 

        Returns:
            vtx (Pointer[Py_ssize_t]): C array containing index of vertices that contribute to constructing the cell.

        """
        cdef float max_val = vtx_filt[0]
        cdef Py_ssize_t j = 0, count = 0

        # find maximum filtration value
        for i in range(1, arr_size):
            if vtx_filt[i] > max_val:
                max_val = vtx_filt[i]
        
        # count how many times max_val occurs
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                count += 1

        cdef Py_ssize_t *vtx = <Py_ssize_t *> malloc(count * sizeof(Py_ssize_t))
        
        # store the index of vertices that have maximum filtration value
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                vtx[j] = neighbor_vtx[i]
                j += 1

        # number of max_val occurences
        num_max[0] = count
        return vtx

    cdef _set_cell_dims(self):
        """
        Sets dimension for all cubes in the cubical complex. Dimensions of vertice, edge, square are 0, 1, 2 respectively. Even rows consist of (vertice, edge, vertice, edge, ..., vertice, edge, vertice) and odd rows consist of (edge, square, edge, square, ..., edge, square, edge).

        Returns:
            _type_: _description_
        """
        dimension = np.zeros([self.cD, self.cH, self.cW], dtype=dtype_float)
        dimension[[i for i in range(self.cD) if i % 2 == 1], :, :] += 1
        dimension[:, [i for i in range(self.cH) if i % 2 == 1], :] += 1
        dimension[:, :, [i for i in range(self.cW) if i % 2 == 1]] += 1
        dimension.setflags(write=False)
        return dimension.flatten()
