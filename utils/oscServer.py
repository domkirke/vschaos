import asyncio
from time import time

from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
import torch, numpy as np
from librosa import stft, istft
from scipy import fft, ifft
from time import  process_time
from ..vaes import AbstractVAE
from ..monitor import visualize_dimred as dr
import dill

from vschaos.vaes import *


# phase reconstruction
phase_rand = {}
def random_static(magnitude):
    if not magnitude.shape[0] in phase_rand.keys():
        phase_rand[magnitude.shape[0]] = np.random.random(magnitude.shape[0]) * 2*np.pi
    return phase_rand[magnitude.shape[0]]

def griffin_lim(transform, iterations = 5):
    if transform.ndim == 1 or transform.shape[0] == 1:
        forward, inverse = fft, ifft
    else:
        forward, inverse = stft, istft
    p = 2 * np.pi * np.random.random_sample(transform.shape) - np.pi
    for i in range(iterations):
        S = transform * np.exp(1j*p)
        inv_p = inverse(S)
        new_p = forward(inv_p)
        new_p = np.angle(new_p)
        # Momentum-modified Griffin-Lim
        p = new_p + (0.99 * (new_p - p))
    return p


def osc_attr(obj, attribute):
    def closure(*args):
        args = args[1:]
        if len(args) == 0:
            return getattr(obj, attribute)
        else:
            return setattr(obj, attribute, *args)
    return closure

class OSCServer(object):
    '''
    Key class for OSCServers linking Python and Max / MSP

    Example :
    >>> server = OSCServer(1234, 1235) # Creating server
    >>> server.run() # Running server

    '''
    # attributes automatically bounded to OSC ports
    osc_attributes = []
    # Initialization method
    def __init__(self, in_port, out_port, ip='127.0.0.1', *args):
        super(OSCServer, self).__init__()
        # OSC library objects
        self.dispatcher = dispatcher.Dispatcher()
        self.client = udp_client.SimpleUDPClient(ip, out_port)
        
        self.init_bindings(self.osc_attributes)
        self.server = osc_server.BlockingOSCUDPServer((ip, in_port), self.dispatcher)
        
        self.in_port = in_port
        self.out_port = out_port
        self.ip = ip

    def init_bindings(self, osc_attributes=[]):
        '''Here we define every OSC callbacks'''
        self.dispatcher.map("/ping", self.ping)
        self.dispatcher.map("/stop", self.stopServer)
        for attribute in osc_attributes:
            print(attribute)
            self.dispatcher.map("/%s"%attribute, osc_attr(self, attribute))

    def stopServer(self, *args):
        '''stops the server'''
        self.client.send_message("/terminated", "bang")
        self.server.shutdown()
        self.server.socket.close()

    def run(self):
        '''runs the SoMax server'''
        self.server.serve_forever()
        
    def ping(self, *args):
        '''just to test the server'''
        print("ping", args)
        self.client.send_message("/fromServer", "pong")
        
    def send(self, address, content):
        '''global method to send a message'''
        self.client.send_message(address, content)

    def print(self, *args):
        print(*args)
        self.send('/print', *args)


def selector_all(dataset):
    return np.arange(dataset.data.shape[0])

def selector_n(dataset, n_points):
    return np.random.permutation(dataset.data.shape[0])[:int(n_points)]



# OSC decorator
def osc_parse(func):
    '''decorates a python function to automatically transform args and kwargs coming from Max'''
    def func_embedding(address, *args):
        t_args = tuple(); kwargs = {}
        for a in args:
            if issubclass(type(a), str):
                if "=" in a:
                    key, value = a.split("=")
                    kwargs[key] = value
                else:
                    t_args = t_args + (a,)
            else:
                t_args = t_args + (a,)
        return func(*t_args, **kwargs)
    return func_embedding


def max_format(v):
    '''Format some Python native types for Max'''
    if issubclass(type(v), (list, tuple)):
        if len(v) == 0:
            return ' "" '
        return ''.join(['%s '%(i) for i in v])
    else:
        return v

def dict2str(dic):
    '''Convert a python dict to a Max message filling a dict object'''
    str = ''
    for k, v in dic.items():
        str += ', set %s %s'%(k, max_format(v))
    return str[2:]

class VAEServer(OSCServer):
    osc_attributes = ['model', 'projection']
    min_threshold = 0
    # model attributes
    def getmodel(self):
        return self._model
    def delmodel(self): del self._model
    def setmodel(self, *args):
       self._model_path = None
       if len(args) == 1:
           if type(args[0]) == str:
               self.print('loading model %s...'%args[0])
               try:
                   loaded_data = torch.load(args[0], map_location='cpu')
               except FileNotFoundError:
                   self.print('file %s not found'%args[0])
                   pass

               self._model = loaded_data['class'].load(loaded_data)
               self._model_path = args[0]
               if 'preprocessing' in loaded_data.keys():
                   self.preprocessing = loaded_data['preprocessing']
                   print(self.preprocessing)
       elif issubclass(type(args[0]), AbstractVAE):
           self._model = args[0]
       self.print('model loaded')
       self.send('/model_loaded', 'bang')
       self.current_projection = None
       self.get_state()
    model = property(getmodel, setmodel, delmodel, "vae model attached to server")

    # projection attributes
    def getprojection(self):
        return self.model.manifolds.get(self.current_projection)
    def delprojection(self):
        if self.current_projection is None:
            raise Exception('[error] tried to delete current projection but is actually empty')
        del self.model.manifolds[self.current_projection]
        self.current_projection = None
    def setprojection(self, *args):
        if args[0] == "none":
            self.current_projection = None
        else:
            assert type(args[0]) == str and args[0] in self.model.manifolds.keys(), "projection %s not found"%args[0]
            self.current_projection = str(args[0])
        self.print('projection set to %s'%self.current_projection)
        self.get_state()
    projection = property(getprojection, setprojection, delprojection)


    def getselector(self): return self._selector
    def delselector(self): raise Exception("VAEServer's cannot be deleted")
    def setselector(self, *args):
        if issubclass(type(args[0]), str):
            self._selector = self.selector_hash[args[0]]
        else:
            self._selector = args[0]
    selector = property(getselector, delselector, setselector)


    def __init__(self, *args, **kwargs):
        self._model = kwargs.get('model')
        self._model_path = None
        self._projections = {}
        self.current_projection = None
        self.phase_reconstruction = None
        self.dataset = kwargs.get('dataset')
        self.preprocessing = kwargs.get('preprocessing')
        self.phase_callback = random_static
        self._selector = selector_all(self.dataset) if self.dataset is not None else None

        super(VAEServer, self).__init__(*args)
        self.get_state()

    def init_bindings(self, osc_attributes=[]):
        super(VAEServer, self).init_bindings(self.osc_attributes)
        self.dispatcher.map('/get_spectrogram', osc_parse(self.get_spectrogram))
        self.dispatcher.map('/add_projection', osc_parse(self.add_projection))
        self.dispatcher.map('/load', osc_parse(self.load_projections))
        self.dispatcher.map('/save', osc_parse(self.save))
        self.dispatcher.map('/get_projections', osc_parse(self.get_projections))
        self.dispatcher.map('/get_state', osc_parse(self.get_state))
        self.dispatcher.map('/add_anchor', osc_parse(self.add_anchor))
        self.dispatcher.map('/load_anchor', osc_parse(self.load_anchor))


    def get_state(self):
        if self.current_projection is not None:
            input_dim = self.projection.dim
        elif self.model is not None:
            input_dim = self._model.pinput['dim']
        else:
            input_dim = 0
        state = {'current_projection': str(self.current_projection),
                 'projections': list(self.model.manifolds.keys()),
                 'input_dim': input_dim}

        state['anchors'] = list(self.model.manifolds[self.current_projection].anchors.keys())

        state_str = dict2str(state)
        self.send('/state', state_str)
        return state


    def get_spectrogram(self, *args, projection=None, filter=True):
        if self._model is None:
           print('[Error] spectrogram requested by not any model loaded')
           self.send('/vae_error', '[Error] spectrogram requested by not any model loaded')
           return

        if projection is None:
            projection = self.projection

        z_input = np.array(list(args))[np.newaxis, :]
        input_dim = self._model.platent[-1]['dim'] if projection is None else projection.dim

        if z_input.shape[1] > input_dim:
            z_input = z_input[:, :input_dim]
        elif z_input.shape[1] < input_dim:
            z_input = np.concatenate([z_input, np.zeros((z_input.shape[0], input_dim - z_input.shape[1]))], axis=1)

        if projection is not None:
            z_input = projection.invert(z_input)

        with torch.no_grad():
            self._model.eval()
            vae_out = self._model.decode(self._model.format_input_data(z_input))

        spec_out = vae_out[0]['out_params'].mean.squeeze().numpy()
        if self.preprocessing:
            spec_out = self.preprocessing.invert(spec_out)
        #phase_out = self.phase_callback(spec_out)

        # spec_out = np.zeros_like(spec_out)
        # id_range = np.random.randint(spec_out.shape[0])
        # spec_out[id_range] = 1

        if filter:
            spec_out[spec_out < self.min_threshold] = 0.
        spec_out = spec_out[1:1024].tolist()
        #phase_out = phase_out[1:].tolist()

        self.send('/current_z', z_input.squeeze().tolist())
        self.send('/spectrogram_mag_out', spec_out)
        #self.send('/spectrogram_phase_out', phase_out)


    def add_projection(self, proj_name, dimred, **kwargs):

        if self._model is None:
            self.print('Please load a model first!')

        if proj_name == "none":
            self.print('projection cannot be named none')

        dimred_method = getattr(dr, dimred)
        if dimred_method is None:
            self.print('projection %s not found'%dimred_method)
        self.print('computing projection %s...'%proj_name)
        if self.preprocessing:
            kwargs['preprocessing'] = self.preprocessing
        self._model.add_manifold(proj_name, self.dataset, dimred_method, **kwargs)

        self.print('projection %s created'%(proj_name))
        self.get_state()


    def get_projections(self):
        self.send('/projections', list(self.model.manifolds.keys()))

    def save(self, path):
        if self._model is None:
            self.print('load a model first!')
            return
        model_manifolds = self._model.manifolds
        torch.save(model_manifolds, path)

    def add_anchor(self, name, *args):
        self.model.manifolds[self.current_projection].add_anchor(name, *args)
        self.get_state()

    def load_anchor(self, name, *args):
        anchor = self.model.manifolds[self.current_projection].anchors[name]
        #self.get_spectrogram(*tuple(np.array(anchor).tolist()))
        self.send('/anchor', anchor)
        self.print('anchor %s loaded'%name)

        # with open(path+'.aego', 'wb') as f:
        #     dill.dump(self._projections, f)
        # self.print('projections saved at %s.aego'%path)

    def load_projections(self, path):
        with open(path, 'rb') as f:
            loaded_projections = torch.load(f)
        for k, v in loaded_projections.items():
            if k in self._projections.keys():
                self.print('[Warning] projection %s replaced'%k)
        self.print('projections loaded from %s'%path)
        self.model.manifolds = loaded_projections
        self.get_state()




