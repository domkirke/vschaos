"""

 Import toolbox       : Audio dataset import

 This file contains the definition of an audio dataset import.

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""

try:
    from matplotlib import pyplot as plt
except:
    import matplotlib 
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    
    
import pdb, torch
import numpy as np
import os
import re

#pdb.set_trace()
from . import data_utils as du
from . import data_generic as generic
from . import data_asynchronous as asyn
import librosa
from ..utils import checklist



"""
###################################
# Initialization functions
###################################
"""


class DatasetAudio(generic.Dataset):    
    """
    Definition of an audio dataset object

    Attributes
    ----------
    types (list)
        list of understood audio types
    importBatchSize (int)
        size of batch during data loading
    transformName (str)
        name of the imported transform
    forceRecompute (bool)
        forces re-computation of transforms


    See also
    --------
    Dataset, RawDatasetAudio, datasetMidi

    """
    def __init__(self, options):
        super(DatasetAudio, self).__init__(options)
        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au', 'npz', 'npy'];
        self.importBatchSize = options.get("importBatchSize") or 64;
        # name of the transform importe
        self.transformName = options.get("transformName") or None;
        self.transformOptions = options.get("transformParameters") or None;
        self.forceRecompute = options.get("forceRecompute") or False;

        # Type of audio-related augmentations
        self.augmentationCallbacks = [];
        # if self.importType == "asynchronous":
        #     self.flattenData = self.flattenDataAsynchronous

    """
    ###################################
    # Import functions
    ###################################
    """
    
    def import_data(self, idList, options, padding=False):
        """
        imports data from corresponding analysis files

        Parameters
        ----------
        options : dict
            in adddition to Dataset options:
                transformName (str) : name of imported transform
                dataDirectory (str) : audio directory of datase
        Returns
        -------
        A new dataset object
        """
        options["transformName"] = options.get("transformName") or self.transformName;
        options["dataDirectory"] = options.get("dataDirectory") or self.dataDirectory;
        options["analysisDirectory"] = options.get("analysisDirectory") or self.analysisDirectory;
        
        # We will create batches of data
        indices = []
        
        # If no idList is given then import all !
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
            if len(indices.shape) == 1:
                indices = indices[np.newaxis]
            
        # Init data
        self.data = []
        new_files = []
        new_metadata = {k: [] for k in self.metadata.keys()}
        new_hash = {f: [] for f in self.files}

        # Parse through the set of batches
        for v in indices:
            curFiles = [None] * v.shape[0]
            for f in range(v.shape[0]):
                curFiles[f] = self.files[int(v[f])]
            curData, curMeta = self.import_audio_data(curFiles, options)

            for f in range(v.shape[0]):
                curData = checklist(curData)
                if issubclass(type(curData[f]), list):
                    self.data.extend(curData[f])
                    new_files = new_files + [self.files[f]]*len(curData[f])
                    for k, v in self.metadata.items():
                        new_metadata[k].extend([v[f]]*len(curData[f]))
                else:
                    self.data.append(curData[f])
                    new_files.append(self.files[f])
                    for k, v in self.metadata.items():
                        new_metadata[k].append(v[f])


        for i, f in enumerate(new_files):
            new_hash[f].append(i)

        self.files = new_files; self.metadata = new_metadata; self.hash = new_hash
        self.files = np.array(self.files)
        self.metadata = {k:np.array(v) for k, v in self.metadata.items()}
        self.clean()

    def list_directory(self, check=True):
        super().list_directory(check=check)
        if check and self.transformName is not None:
            self.check_input_transforms()

    def clean(self):
        """
        cleans empty data
        """
        if issubclass(type(self.data), list):
            for i in reversed(range(len(self.data))):
                if self.data[i] is None:
                    del self.data[i]
                elif issubclass(type(self.data[i]), list):
                    if len(self.data[i]) == 0:
                        del self.data[i]
                        f = self.files[i]
                        del self.files[i]
                        del self.hash[f]
                        for k in self.metadata.keys():
                            if issubclass(type(self.metadata[k]), list):
                                del self.metadata[k][i]
                            elif issubclass(type(self.metadata[k]), np.ndarray):
                                self.metadata[k] = np.delete(self.metadata[k], i)

    def import_audio_data(self, curBatch, options):
        """
        import the audio data and metadata of the files in curBatch

        Parameters
        ----------
        curBatch : list(str)
            list of the files to import
        options : dict

        Returns
        -------
        data : list(np.ndarray)
            imported data
        metadata : dict
            metadata dictionary
        """
        def load(path):
            if os.path.splitext(path)[1] == ".npy":
                return np.load(path)
            elif os.path.splitext(path)[1] == ".npz":
                return np.load(path)['arr_0']

        dataPrefix = options.get('dataPrefix')
        dataDirectory = options.get('dataDirectory') or dataPrefix+'/data' or ''
        analysisDirectory = options.get('analysisDirectory') 
        if analysisDirectory is None: 
            try:
                analysisDirectory = options.get('dataPrefix')
                analysisDirectory += '/analysis'
            except TypeError:
                print('[Error] Please specify an analysis directory to import audio data')
             
        transformName= options.get('transformName')
        finalData = []
        finalMeta = []
        for f in curBatch:
            if transformName is None:
                finalData.append(librosa.load(f)[0])
                finalMeta.append(0)
            else:
                curAnalysisFile = os.path.splitext(re.sub(dataDirectory, analysisDirectory+'/'+transformName, f))[0]
                files = []; idx = 0;
                name, ext = os.path.splitext(curAnalysisFile)
                if os.path.exists(name+'.npz'):
                    files.append(name+'.npz')
                elif os.path.exists(name+'.npy'):
                    files.append(name+'.npy')
                else:
                    while os.path.exists(name+'_%d'%idx+'.npz') or os.path.exists(name+'_%d'%idx+'.npy'):
                        if os.path.exists(name+'_%d'%idx+'.npz'):
                            files.append(name+'_%d'%idx+'.npz')
                        else:
                            files.append(name+'_%d'%idx+'.npy')
                        idx += 1
                if len(files) == 0:
                    print('[Warning] did not found file %s'%curAnalysisFile)
                    continue
                curAnalysisFile = files

                if issubclass(type(curAnalysisFile), list):
                    finalData.append([load(c) for c in curAnalysisFile])
                    finalMeta.append([0]*len(curAnalysisFile))
                else:
                    finalData.append(load(curAnalysisFile))
                    finalMeta.append(0)

        if os.path.isfile("%s/%s/transformOptions.npy"%(analysisDirectory, transformName)):
            self.transformOptions = np.load("%s/%s/transformOptions.npy"%(analysisDirectory, transformName), allow_pickle=True)[None][0]
        return finalData, finalMeta


    def check_input_transforms(self):
        """
        filter files that do not have the corresponding analysis files
        """

        for i in reversed(range(len(self.files))):
            analysis_path = [os.path.splitext(re.sub(self.dataDirectory, self.analysisDirectory+'/'+self.transformName, self.files[i]))[0]+'.npy',
                             os.path.splitext(re.sub(self.dataDirectory, self.analysisDirectory+'/'+self.transformName, self.files[i]))[0]+'.npz']
            tests = []
            for a_tmp in analysis_path:
                try:
                    a_tmp = analysis_path[analysis_path.index(a_tmp)]
                    fiD_test = open(a_tmp)
                    fiD_test.close()
                    tests.append(True)
                except FileNotFoundError:
                    tests.append(False)
                    pass
            if not True in tests:
                del self.files[i]
        self.hash = {self.files[i]:i for i in range(len(self.files))}


    """
    ###################################
    # Obtaining transform set and options
    ###################################
    """
    
    def get_transforms(self):
        """
        Transforms (and corresponding options) available

        :returns:
        transform_list: list(str)
            list of available transforms
        default_options : dict
            dictionary of available options

        """
        # List of available transforms
        transformList = [
            'raw',                # raw waveform
            'stft',               # Short-Term Fourier Transform
            'mel',                # Log-amplitude Mel spectrogram
            'mfcc',               # Mel-Frequency Cepstral Coefficient
#            'gabor',              # Gabor features
            'chroma',             # Chromagram
            'cqt',                # Constant-Q Transform
            'gammatone',          # Gammatone spectrum
            'dct',                # Discrete Cosine Transform
#            'hartley',            # Hartley transform
#            'rasta',              # Rasta features
#            'plp',                # PLP features
#            'wavelet',            # Wavelet transform
#            'scattering',         # Scattering transform
#            'cochleogram',        # Cochleogram
            'strf',               # Spectro-Temporal Receptive Fields
            'csft',                 # Cumulative Sampling Frequency Transform
            'modulation',          # Modulation spectrum
            'nsgt',               # Non-stationary Gabor Transform
            'nsgt-cqt',               # Non-stationary Gabor Transform (CQT scale)
            'nsgt-mel',               # Non-stationary Gabor Transform (Mel scale)
            'nsgt-erb',               # Non-stationary Gabor Transform (Mel scale)
            'strf-nsgt',              # Non-stationary Gabor Transform (STRF scale)
        ];
                
        # List of options
        transformOptions = {
            "debugMode":0,
            "resampleTo":22050,
            "downsampleFactor":0,
            "targetDuration":0,
            "winSize":2048,
            "hopSize":1024,
            #"nFFT":2048,
            # Normalization
            "normalizeInput":False,
            "normalizeOutput":False,
            "equalizeHistogram":False,
            "logAmplitude":False,
            #Raw features
            "chunkSize":8192,
            "chunkHop":4096,
            "grainSize":1024,
            "grainOverlap":512,
            # Phase
            "removePhase":False,
            "concatenatePhase":False,
            # Mel-spectrogram
            "minFreq":30,
            "maxFreq":11000,
            "nbBands":128,
            # Mfcc
            "nbCoeffs":13,
            "delta":0,
            "dDelta":0,
            # Gabor features
            "omegaMax":'[pi/2, pi/2]',
            "sizeMax":'[3*nbBands, 40]',
            "nu":'[3.5, 3.5]',
            "filterDistance":'[0.3, 0.2]',
            "filterPhases":'{[0, 0], [0, pi/2], [pi/2, 0], [pi/2, pi/2]}',
            # Chroma
            "chromaWinSize":2048,
            # CQT
            "cqtBins":360,
            "cqtBinsOctave":60,
            "cqtFreqMin":64,
            "cqtFreqMax":8000,
            "cqtGamma":0.5,
            # Gammatone
            "gammatoneBins":64,
            "gammatoneMin":64,
            # Wavelet
            "waveletType":'\'gabor_1d\'',
            "waveletQ":8,
            # Scattering
            "scatteringDefault":1,
            "scatteringTypes":'{\'gabor_1d\', \'morlet_1d\', \'morlet_1d\'}',
            "scatteringQ":'[8, 2, 1]',
            "scatteringT":8192,
            # Cochleogram
            "cochleogramFrame":64,        # Frame length, typically, 8, 16 or 2^[natural #] ms.
            "cochleogramTC":16,           # Time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
            "cochleogramFac":-1,          # Nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
            "cochleogramShift":0,         # Shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
            "cochleogramFilter":'\'p\'',      # Filter type ('p' = Powen's IIR, 'p_o':steeper group delay)
            # STRF
            "strfFullT":0,                # Fullness of temporal margin in [0, 1].
            "strfFullX":0,                # Fullness of spectral margin in [0, 1].
            "strfBP":0,                   # Pure Band-Pass indicator
            "strfRv": np.power(2, np.linspace(0, 5, 5)),     # rv: rate vector in Hz, e.g., 2.^(1:.5:5).
            "strfSv": np.power(2, np.linspace(-2, 3, 6)),    # scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
            "strfMean":0,                  # Only produce the mean activations
            "csftDensity":512,
            "csftNormalize":True
        }
        return transformList, transformOptions;
       
    def __dir__(self):
        tmpList = super(DatasetAudio, self).__dir__()
        return tmpList + ['importBatchSize', 'transformType', 'matlabCommand']
    

    """
    ###################################
    # Plotting functions
    ###################################
    """

    # def plotExampleSet(self, setData, labels, task, ids):
    #     fig = plt.figure(figsize=(12, 24))
    #     ratios = np.ones(len(ids))
    #     fig.subplots(nrows=len(ids),ncols=1,gridspec_kw={'width_ratios':[1], 'height_ratios':ratios})
    #     for ind1 in range(len(ids)):
    #         ax = plt.subplot(len(ids), 1, ind1 + 1)
    #         if (setData[ids[ind1]].ndim == 2):
    #             ax.imshow(np.flipud(setData[ids[ind1]]), interpolation='nearest', aspect='auto')
    #         else:
    #             tmpData = setData[ids[ind1]]
    #             for i in range(setData[ids[ind1]].ndim - 2):
    #                 tmpData = np.mean(tmpData, axis=0)
    #             ax.imshow(np.flipud(tmpData), interpolation='nearest', aspect='auto')
    #         plt.title('Label : ' + str(labels[task][ids[ind1]]))
    #         ax.set_adjustable('box-forced')
    #     fig.tight_layout()
    #     
    # def plotRandomBatch(self, task="genre", nbExamples=5):
    #     setIDs = np.random.randint(0, len(self.data), nbExamples)
    #     self.plotExampleSet(self.data, self.metadata, task, setIDs)
        

           
    """
    ###################################
    # Transform functions
    ###################################
    """
    
    def compute_transforms(self, transformTypes, transformParameters, transformNames=None, idList=None, padding=False, forceRecompute=False, verbose=False):
        """
        .. function:: compute_transforms(transfromTypes, transformParameters[, transformNames=None, idList=None, padding=None])

        Compute the transforms corresponding to the imported files.
        :param transformTypes: Types of the transforms to compute
        :type transformTypes: list(str)
        :param transformParameters: Parameters of the corresponding transforms
        :type transformParameters: list(dict):
        :param transformNames: Names of the corresponding transforms
        :param idList: if given, only compute transforms for the given IDs
        :type idList: list(int)

        :returns:
        transform_list: list(str)
            list of available transforms
        default_options : dict
            dictionary of available options

        """
        dataDirectory = self.dataDirectory or self.dataPrefix+'/data'
        analysisDirectory = self.analysisDirectory or self.dataPrefix+'/analysis'
        if transformNames is None:
            transformNames = transformTypes
        if len(transformNames)!=len(transformTypes):
            raise Exception('please give the same number of transforms and names')
            
        if not issubclass(type(transformTypes), list):
            transformTypes = [transformTypes]
        if not issubclass(type(transformParameters), list):
            transformParameters = [transformParameters]
        if not issubclass(type(transformNames), list):
            transformNames = [transformNames]
        
        # get indices to compute
        if (idList is None):
            indices = np.linspace(0, len(self.files) - 1, len(self.files))
            if padding:
                indices = np.pad(indices, (0, self.importBatchSize - (len(self.files) % self.importBatchSize)), 'constant', constant_values=0)
                indices = np.split(indices, len(indices) / self.importBatchSize)
            else:
                indices = [indices]
        else:
            indices = np.array(idList)
                        
        if not os.path.isdir(analysisDirectory):
            os.makedirs(analysisDirectory)
            
        for i in range(len(transformTypes)):
            current_transform_dir = analysisDirectory+'/'+transformNames[i]
            if not os.path.isdir(current_transform_dir):
                os.makedirs(current_transform_dir)
            for v in indices:
                curFiles = [None] * v.shape[0]
                for f in range(v.shape[0]):
                    curFiles[f] = self.files[int(v[f])]
                makeAnalysisFiles(curFiles, transformTypes[i], transformParameters[i], dataDirectory, analysisDirectory+'/'+transformNames[i], transformName=transformNames[i], forceRecompute=forceRecompute, verbose=verbose)
                
            # save transform parameters
            transformParameters[i]['transformType'] = transformTypes[i]
            np.save(current_transform_dir+'/transformOptions.npy', transformParameters[i])

    def retrieve(self, idx):
        dataset = super(DatasetAudio, self).retrieve(idx)
        dataset.transformOptions = self.transformOptions
        return dataset



class OfflineDatasetAudio(DatasetAudio):
    """
    :class:`DatasetAudio` that only imports data when directly inquired. This class is based on the
    :class:`OfflineDataList`, that fints a regular list but only loads data when __getitem__ is called. This class
    is suitable for big datasets that cannot be directly imported to RAM.
    """
    def __init__(self, *args, entry_class=asyn.OfflineEntry, **kwargs):
        """
        In addition to DatasetAudio init arguments:
            :param entry_class: subclass of :class:`data_asynchronous.OfflineEntry` used for asynchrounous import
            :type entry_class: type
        """
        super(OfflineDatasetAudio, self).__init__(*args, **kwargs)
        self.entry_class = entry_class

    def __len__(self):
        if hasattr(self.data, "shape"):
            # if data is np.ndarray or tensor, return first dimensions
            # can also be OfflineDataList, so check is shape is None
            return self.data.shape[0] if self.data.shape else len(self.data)
        else:
            if self.has_sequences:
                return len(self.data)
            else:
                return self.data[0].shape[0]

    def import_audio_data(self, curBatch, options):
        dataPrefix = options.get('dataPrefix')
        dataDirectory = options.get('dataDirectory') or dataPrefix+'/data' or ''
        analysisDirectory = options.get('analysisDirectory')
        if analysisDirectory is None:
            try:
                analysisDirectory = options.get('dataPrefix')
                analysisDirectory += '/analysis'
            except TypeError:
                print('[Error] Please specify an analysis directory to import audio data')

        transformName= options.get('transformName')
        padded = options.get('padded')
        finalData = []
        finalMeta = []
        for f in curBatch:
            curAnalysisFile = re.sub(dataDirectory, analysisDirectory+'/'+transformName, f)
            curAnalysisFile = ".".join(curAnalysisFile.split(".")[:-1])
            if os.path.exists(curAnalysisFile+'.npy'):
                curAnalysisFile = curAnalysisFile + '.npy'
            elif os.path.exists(curAnalysisFile+'.npz'):
                curAnalysisFile = curAnalysisFile + '.npz'
            else:
                files = []; idx = 0;
                name=curAnalysisFile
                while os.path.exists(name+'_%d'%idx+'.npz') or os.path.exists(name+'_%d'%idx+'.npy'):
                    if os.path.exists(name+'_%d'%idx+'.npz'):
                        files.append(name+'_%d'%idx+'.npz')
                    else:
                        files.append(name+'_%d'%idx+'.npy')
                    idx += 1
                if len(files) == 0:
                    print('[Warning] did not found file %s'%curAnalysisFile)
                    continue
                curAnalysisFile = files


            if issubclass(type(curAnalysisFile), list):
                finalData.append(asyn.OfflineDataList([self.entry_class(c) for c in curAnalysisFile], padded=padded))
                finalMeta.append([0]*len(curAnalysisFile));
            else:
                finalData.append(self.entry_class(curAnalysisFile))
                finalMeta.append(0);

        return finalData, finalMeta


    def flatten_data(self, selector=lambda x: x, window=None, window_overlap=0.5, padded=False):
        # initialize
        newData = []
        newMetadata = {}
        for k, v in self.metadata.items():
            newMetadata[k] = []
        newFiles = []
        newPartitions = {k:[] for k in self.partitions.keys()}
        revHash = {}
        # new hash from scratch
        newHash = {k:[] for k in self.hash.keys()}
        # filter dataset
        idx = 0
        for i in range(len(self.data)):
            # update minimum content shape
            chunk_to_add = selector(self.data[i].entries)
            newData.extend(chunk_to_add)
            for k, _ in newMetadata.items():
                newMetadata[k].extend([self.metadata[k][i]]*len(chunk_to_add))
            newFiles.extend([self.files[i]]*len(chunk_to_add))
            current_idxs = list(range(idx, idx + len(chunk_to_add)))
            newHash[self.files[i]].extend(current_idxs)
            for name, part in self.partitions.items():
                if i in part or self.files[i] in part:
                    if type(self.partitions[name][0])==int:
                        newPartitions[name].extend(current_idxs)
                    else:
                        newPartitions[name].append(self.files[i])
            idx += len(chunk_to_add)

        self.data = asyn.OfflineDataList(newData, dtype=newData[0]._dtype, padded=padded)
        self.metadata = newMetadata
        for k,v in newMetadata.items():
            newMetadata[k] = np.array(v)
        self.files = newFiles
        self.hash = newHash
        self.revHash = revHash
        self.partitions = newPartitions

    def load_offline_entries(self, offline_entries):
        self.data = np.load(offline_entries)


    def save_offline_entries(self, out=None, options={}):
        out = out if out else options.get('analysisDirectory', self.analysisDirectory)
        np.save(out+'/offline_calls.npy', self.data)




"""
###################################
# External functions for transfroms and audio imports
###################################
"""       

def makeAnalysisFiles(curBatch, transformType, options, oldRoot, newRoot, transformName=None, backend='python', forceRecompute=False, verbose=False):
    from .signal.transforms import computeTransform
    transformName = transformName or transformType;
                    
    # Parse through the set of batches
    curAnalysisFiles = [None] * len(curBatch)
    audioList = [None] * len(curBatch)
    curIDf = 0
    
    # Check which files need to be computed
    for i in range(len(curBatch)):
        curFile = curBatch[i]
        analysisName = os.path.splitext(curFile.replace(du.esc(oldRoot), newRoot))[0]
        try:
            fIDTest = open(analysisName, 'r')
        except IOError:
            fIDTest = None
        if ((fIDTest is None) or (forceRecompute == True)):
            audioList[curIDf] = curFile 
            curAnalysisFiles[curIDf] = analysisName
            curIDf = curIDf + 1
        else: 
            fIDTest.close()
    audioList = audioList[:curIDf]
    curAnalysisFiles = curAnalysisFiles[:curIDf]
    
    # Compute transform of files that have not been computed yet
    if (len(audioList) > 0):
        if verbose:
            print("* Computing transforms ...")        
        for i, target_file in enumerate(curAnalysisFiles):
            current_transforms = computeTransform([audioList[i]], transformType, options, out=curAnalysisFiles[i])

            
        
