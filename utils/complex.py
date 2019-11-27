import torch, pdb, numpy as np

eps = torch.tensor(1e-12)

def complex_op(self, func):
    def complex_wrapper(*args, **kwargs):
        return ComplexTensor(func(*args, **kwargs))
    return complex_wrapper


class ComplexTensor(torch.Tensor):
    def __new__(cls, data, imag=None, batch=True, complex_dim=None, *args, **kwargs):

        #device = torch.device(data.device)
        if imag is None:
            if data.ndimension() == 1:
                return data
        else:
            # imag = imag.to(device=device)
            data = torch.cat([data, imag], dim=int(batch))

        return super().__new__(cls, data, *args, **kwargs)
        #new_tensor = torch.Tensor._make_subclass(ComplexTensor, data, da#ta.requires_grad)
        # ComplexTensor.__init__(new_tensor, data, batch=batch, complex_dim=complex_dim)
        #new_tensor.to(data.device)
        return new_tensor

    def __repr__(self):
        return 'complex tensor (dim = %d) real : '%self.complex_dim + self.real.__repr__() \
               + '\n\timag: ' + self.imag.__repr__()

    def __init__(self, data=None, imag=None, batch=True, complex_dim=None, device=None, **kwargs):
        print('coucou')
        if device:
            self.device = device
            data = data.to(device=device)
            if imag:
                imag = imag.to(device=device)
            
        self.batch = batch
        if complex_dim is None:
            if self.batch:
                self.complex_dim = 1
            else:
                self.complex_dim = 0
        else:
            self.complex_dim = complex_dim
        size = list(self.shape)
        size[self.complex_dim] = int(size[self.complex_dim] / 2)
        self.rshape = torch.Size(size)
        # if self.shape[self.complex_dim] % 2 == 1:
        #     pdb.set_trace()
        assert self.shape[self.complex_dim] % 2 == 0

        self._real = None; self._imag = None
        self._angle = None; self._radius = None

    def is_squared(self):
        if self.real_ndim() == 2:
            if self.batch:
                return self.rshape[1] == self.rshape[2]
            else:
                return self.rshape[0] == self.rshape[1]

    def real_ndim(self):
        if self.batch:
            return len(self.shape) -1
        else:
            return len(self.shape)

    @property
    def real(self):
        ids_range = torch.tensor(range(int(self.shape[self.complex_dim]/2)), device=self.device)
        return self.index_select(self.complex_dim, ids_range)

    @property
    def imag(self):
        ids_range = torch.tensor(range(int(self.shape[self.complex_dim]/2), self.shape[self.complex_dim]), device=self.device)
        return self.index_select(self.complex_dim, ids_range)

    @property
    def radius(self):
        return torch.sqrt(self.real**2 + self.imag**2 + eps)

    @property
    def angle(self):
        return torch.atan2(self.imag+eps, self.real+eps)

    def disentangle(self):
        if self.ndimension() - int(self.batch) == 1:
            return self
        else:
            return torch.cat((torch.cat((self.real, -self.imag), 2), torch.cat((self.imag, self.real), 2)), 1)

    def conj(self):
        return ComplexTensor(self.real, -self.imag)

    def transpose(self, dim0, dim1):
        if dim0 == self.complex_dim or dim1 == self.complex_dim:
            raise IndexError("transpose(%d, %d): can transpose dim %d"%(dim0, dim1, self.complex_dim))
        return ComplexTensor(self.real.transpose(dim0, dim1), self.real.transpose(dim0, dim1))

    def hermitian(self):
        return ComplexTensor(self.real.t(), -self.imag.t())

    def det(self):
        full_matrix = self.disentangle()
        if not self.batch:
            return torch.sqrt(torch.det(full_matrix[i]))
        else:
            dets = torch.zeros(full_matrix.shape[0], device=full_matrix.device, requires_grad=full_matrix.requires_grad)
            for i in range(dets.shape[0]):
                dets[i] = torch.sqrt(torch.det(full_matrix[i]))
            return dets

    def unsqueeze(self, dim=None):
        if dim is None:
            dim = 1 if self.batch else 0
        if dim < 0:
            dim = len(self.shape) + dim + 1

        new_cdim = self.complex_dim
        if dim <= new_cdim:
            new_cdim += 1
        return ComplexTensor(super(ComplexTensor, self).unsqueeze(dim), complex_dim=new_cdim)


def bcmm(u, v):
    u_dis = u.disentangle()
    v_dis = v.disentangle()
    if u.real_ndim() == 1:
        u_dis = u_dis.unsqueeze(1)
    if v.real_ndim() == 1:
        v_dis = v_dis.unsqueeze(-1)

    mult_result = torch.bmm(u_dis, v_dis)
    return ComplexTensor(mult_result.squeeze())


def cinv(matrix):
    assert(matrix.is_squared())
    real_inv = torch.inverse(matrix.real() + torch.bmm(torch.bmm(matrix.imag(), torch.inverse(matrix.real())), matrix.imag()))
    imag_inv = -torch.inverse(matrix.imag() + torch.bmm(torch.bmm(matrix.real(), torch.inverse(matrix.imag())), matrix.real()))
    return ComplexTensor(real_inv, imag_inv)



