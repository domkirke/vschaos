import os, numpy as np, torch
from .data_generic import Dataset


def default_symbolic(file):
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    ext = os.path.splitext(file)[1]
    if ext in ['.npy', '.npz']:
        data = np.load(file, allow_pickle=True)
    elif ext == '.t7':
        data = torch.load(file)
    return data



class SymbolicDataset(Dataset):

    def __init__(self, options):
        super().__init__(options)
        self.types = ['npz', 'npy', 't7']

    def importData(self, ids, options):
        options = options or {}
        symbolic_callback = options.get('callback', default_symbolic)
        callback_args = options.get('callback_args', {})
        ids = ids or range(len(self.files))

        data = []
        for i in range(len(ids)):
            data.append(symbolic_callback(self.files[i], **callback_args))

        self.data = data




