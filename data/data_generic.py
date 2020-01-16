"""

    The ``Generic dataset import`` module
    ========================
 
    This file contains the basic definitions for any type of dataset.
    Importing a dataset usually relies on the same set of functions
        * Find files in a directory
        * List files from a metadata file
        * Import data from the files
        * Perform transform
        * Data augmentation

    Currently implemented
    ---------------------
    __init__(options)
        Class constructor
    retrieve(self, idx)
        retrives given indices or partition
    listDirectory()
        Fill the list of files from a direct recursive path of the dataDirectory attribute
    listInputFile(inputFile)
        List informations over a dataset (based on a given inputs.txt file)
    checkInputTransforms()
        Remove files that do not have the corresponding analysis files
    retrieve_tasks()
        Retrieve the tasks of the current dataset
    importMetadata(fileName, task, callback)
        Import metadata from filename, task and callback
    importMetadataTasks()
        Perform all metadata imports based on pre-filled tasks
    import_data()
        Import all the data files directly
    import_dataAsynchronous(dataIn, options)
        Import the data files asynchronously
    create_subsets_partitions(task)
        Create subsets of data from the partitions
    construct_partitions(tasks, partitionNames, partitionPercent, balancedClass, equalClass)
        Construct a random/balanced partition set for each dataset
    construct_partitions_files(self, partitionNames, partitionFiles):
        Constructing partitions from a given set of files
    augmentDataAsynchronous(dataIn, options):
        Perform balanced and live data augmentation for a batch
    get(i)
        Access a single element in the dataset 
    create_batches(partition, batchSize, balancedTask)
        Create a set of batches balanced in terms of classes
    flatten_data(self, selector_fn, window=None, window_overlap=0.5)
        Flatten the entire data matrix
    window(window_size, axis=1)
        window along given dimentions
    filter_set(datasetIDx, currentMetadata, curSep)
        Filter the dataset depending on the classes present
    remove_files(n_files, shuffle=True)
        just keeps n_files in current data
    filter_files(files)
        just keep files contained in files array
    save(location, **kwargs)
        Pickles the dataset at given path
    load(location)
        Load the dataset at given path

    
    Comments and issues
    -------------------
    None for the moment

    Contributors
    ------------
    Philippe Esling (esling@ircam.fr)
    Axel Chemla--Romeu-Santos (chemla@ircam.fr)
    
"""

import re, pdb, copy, torchvision, dill
import numpy as np
from os import path
import os
# Package-specific imports
from . import data_metadata
from .data_metadata import metadataCallbacks
from . import data_utils
import torch.utils.data
from .data_asynchronous import OfflineDataList, OfflineEntry
from ..utils import checklist, GPULogger


class Dataset(torch.utils.data.Dataset):    
    """ 
    
    Definition of a basic dataset object
    
    Note
    ----
    This class should be avoided, check for more specialized classes

    Attributes
    ----------
    dataDirectory : str
        Path to the current dataset
    analysisDirectory : str
        Path to the analysis (transforms if needed)
    metadataDirectory : str
        Path to the metadata
    dataPrefix : str
        Path to relative analysis of metadata
    importType : str
        Type of importing (asynchronous, direct)
    importCallback : function pointer
        Function pointer for importing data
    types : list of str
        List of all accepted filetypes
    tasks : list of str
        List of the supervised tasks for which to find metadata
    taskCallback : list of function pointers
        List of functions to import the metadata related to each task
    partitions : list of numpy arrays
        Sets of partitions for valid, test or train style of splits
    verbose : bool
        Activate verbosity to print all information
    forceUpdate : bool
        Force to recompute transforms on all files
    checkIntegrity : bool
        Check that all files are correct
    augmentationCallbacks : list of function pointers
        Set of data augmentation functions
    hash : dict
        Dictionnary linking files to their data indices
    files : list of str
        Sets of paths to the dataset files
    classes : dict of numpy array
        Dict of class names along with their indices
    metadata : dict of numpy array
        Dict of metadatas for all tasks
    labels : dict of numpy array
        Lists of labels (related to classes indices)
    data : list of numpy arrays
        Set of the data for all files
    metadataFiles : list of str
        List of all files for metadata
        
    See also
    --------
    datasetAudio, rawDatasetAudio, datasetMidi

    """

        
    def __init__(self, options):
        """ 
        Class constructor 
        
        Parameters
        ----------
        options : dict
            native options:
                dataPrefix (str) : data root of dataset
                dataDirectory (str) : audio directory of dataset (default: dataPrefix + '/data')
                analysisDirectory (str) : transform directory of dataset (default: dataPrefix + '/analysis')
                metadataDirectory (str) : metadata directory of dataset (default: dataPrefix + '/metadata')
                tasks [list(str)] : tasks loaded from the dataset
                taskCallback[list(callback)] : callback used to load metadata (defined in data_metadata)
                verbose (bool) : activates verbose mode
                forceUpdate (bool) : forces updates of imported transforms
                checkIntegrity (bool) : check integrity of  files

        Returns
        -------
        A new dataset object
 
        Example
        -------
        """
        # Directories
        self.dataPrefix = options.get("dataPrefix") or ''
        self.dataDirectory = options.get("dataDirectory") or (self.dataPrefix + '/data')
        self.analysisDirectory = options.get("analysisDirectory") or (self.dataPrefix + '/analysis')
        self.metadataDirectory = options.get("metadataDirectory") or (self.dataPrefix + '/metadata')
        
        # Type of import (direct, asynchronous)
        # self.importType = options.get("importType") or 'direct'
        self.importCallback = options.get("importCallback") or {}
        self.has_sequences = False
        # data padding in case of variable length inputs
        self.padded_dims = set()
        self.padded_lengths = {}

        # Accepted types of files
        self.types = options.get("types") or ['mp3', 'wav', 'wave', 'aif', 'aiff', 'au']
        
        # Tasks to import
        self.tasks = options.get("tasks", [])
        self.taskCallback = [None] * len(self.tasks)
        
        # Partitions in the dataset
        self.partitions = {}
        self.partitions_files = {}
        
        # Filters in the dataset
        # Properties of the dataset
        self.verbose = options.get("verbose") or False
        self.forceUpdate = options.get("forceUpdate") or False
        self.checkIntegrity = options.get("checkIntegrity") or False;
        
        # Augmentation callbacks (specific to the types)
        self.augmentationCallbacks = [];
        self.hash = {}
        self.files = []
        self.classes = {}
        self.metadata = {} 
        self.labels = []
        self.data = []
        self.metadataFiles = [None] * len(self.tasks)
        for t in range(len(self.tasks)):
            self.taskCallback[t] = (options.get("taskCallback") and options["taskCallback"][t]) or self.retrieve_callback_from_path(self.metadataDirectory, self.tasks[t]) or metadataCallbacks["default"] or []
            self.metadataFiles[t] = (options.get("metadataFiles") and options["metadataFiles"][t]) or self.metadataDirectory + '/' + self.tasks[t] + '/metadata.txt' or self.metadataDirectory + '/metadata.txt'
        self.drop_tasks = None
        if len(self.tasks) > 0:
            self.drop_tasks = self.tasks



    def __getitem__(self, item):
        """
        __getitem__ method of Dataset. Is used when combined with the native torch data loader.

        Parameters
        ----------
        options : dict
            native options:
                dataPrefix (str) : data roo<t of dataset
                dataDirectory (str) : audio directory of dataset (default: dataPrefix + '/data')
                analysisDirectory (str) : transform directory of dataset (default: dataPrefix + '/analysis')
                metadataDirectory (str) : metadata directory of dataset (default: dataPrefix + '/metadata')
                tasks [list(str)] : tasks loaded from the dataset
                taskCallback[list(callback)] : callback used to load metadata (defined in data_metadata)
                verbose (bool) : activates verbose mode
                forceUpdate (bool) : forces updates of imported transforms
                checkIntegrity (bool) : check integrity of  files

        Returns
        -------
        A new dataset object

        Example
        -------
        """
        # Retrieve data
        if issubclass(type(self.data), (list, tuple)):
            if self.has_sequences:
                data = self.data[item]
            else:
                data = [self.data[i][item] for i in range(len(self.data))]
        else:
            data = self.data[item]
        # Pad data in case
        if len(self.padded_dims)!=0:
            if type(data) == list:
                data = [self._get_padded_data(d, self.padded_dims, self.padded_lengths)[np.newaxis] for d in data]
                data = np.concatenate(data, axis=0)
            else:
                data = self._get_padded_data(data, self.padded_dims, self.padded_lengths)
        metadata = []
        # Get corresponding metadata
        if self.drop_tasks:
            metadata = {t:self.metadata[t][item] for t in self.drop_tasks}
        return data, metadata

    def __setitem__(self, idx):
        raise NotImplementedError()

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


    def _get_padded_data(self, data, padded_dims, padded_lengths):
        paddings = [(0,0)]*len(data.shape)
        for d in padded_dims:
            if data.shape[d-1] < padded_lengths[d]:
                paddings[d-1] = (0, self.padded_lengths[d]-data.shape[d-1])
                data = np.pad(data, tuple(paddings), 'constant')
            else:
                data = np.take(data, np.arange(padded_lengths[d]), axis=d-1)
        return data

    def drop_task(self, tasks):
        """the :py:func:`drop_tasks` function tells the dataset which metadata is returned when using the __getitem__ method.
        :param list tasks: list of the tasks to drop during data import (list of str)
        """
        tasks = checklist(tasks)
        for t in tasks:
            if t not in self.tasks:
                raise Exception('the task %s is not present in the dataset'%t)
        self.drop_tasks = list(tasks)

    def get_ids_from_files(self, files, translate=True):
        """returns the data ids corresponding to a given list of files.
        :param list files: list of desired files.
        :param bool translate: does the object translate the files to the actual data roor (default: True)"""
        ids = []
        if translate:
            files = self.translate_files(files)
        for f in files:
            ids.extend(self.hash[f])
        return np.array(ids)

    def get_ids_from_class(self, meta_id, task, ids=None, exclusive=False):
        """return the data ids corresponding to a given class, for a given task.
        :param meta_id: desired class ids
        :type meta_id: int, list, np.ndarray
        :param str task: target task
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        """
        current_metadata = self.metadata.get(task)
        assert current_metadata is not None, "task %s not found"%task
        if ids is None:
            ids = range(len(current_metadata))
        valid_ids = list(filter(lambda x: current_metadata[x] is not None and x in ids, range(len(current_metadata))))
        meta_id = checklist(meta_id)
        ids = set(valid_ids) if exclusive else set()
        for m in meta_id:
            if exclusive:
                ids = set(filter(lambda x: current_metadata[x] == m or m in checklist(current_metadata[x]), valid_ids)).intersection(ids)
            else:
                ids = set(filter(lambda x: current_metadata[x] == m or m in checklist(current_metadata[x]), valid_ids)).union(ids)
        return ids

    def retrieve(self, idx):
        """
        returns a sub-dataset from the actual one. If the main argument is a string, then returns the sub-dataset of the
        corresponding partition. Otherwise, it has to be a valid list of indices.
        :param idx: list of indices or partition
        :type idx: str, list(int), np.ndarray
        :return: the corresponding sub-dataset
        :rtype: :py:class:`Dataset`
        """

        if type(idx) == str:
            # retrieve the corresponding partition ids
            if not idx in self.partitions.keys():
                raise IndexError('%s is not a partition of current dataset'%idx)
            idx = self.partitions[idx]
            if type(idx[0]) == str or type(idx[0]) == np.str_:
                # if the partition are files, get ids from files
                idx = self.get_ids_from_files(idx)

        # create a new dataset
        newDataset = type(self)(self.get_options_dict())

        if len(self.data) > 0:
            if type(self.data) == list:
                #newDataset.data = np.array(self.data)[idx]
                #if self.has_sequences:
                newDataset.data = [self.data[i] for i in idx]
                #else:
                #    newDataset.data = [self.data[i][idx] for i in range(len(self.data))]
            elif type(self.data) == OfflineDataList:
                newDataset.data = self.data.take(idx)
            else:
                #pdb.set_trace()
                newDataset.data = self.data[idx]
        if self.metadata != {}:
            newDataset.metadata = {k:v[idx] for k,v in self.metadata.items()}
        if self.classes != {}:
            newDataset.classes = self.classes
        if len(self.files) != 0:
            newDataset.files = np.array(self.files)[idx].tolist()
        if self.hash != {}:
            newDataset.hash = {newDataset.files[i]:checklist(i) for i in range(len(newDataset.files))}
        if self.augmentationCallbacks != []:
            newDataset.augmentationCallbacks = self.augmentationCallbacks
        if self.labels != []:
            newDataset.labels = self.labels
        newDataset.padded_lengths = self.padded_lengths
        newDataset.padded_dims = self.padded_dims
        newDataset.drop_tasks = self.drop_tasks
        newDataset.has_sequences = self.has_sequences
        #newDataset.transformOptions = self.transformOptionclass_ids)s
            
        if len(self.partitions) != 0:
            newDataset.partitions = {}
            for name, i in self.partitions.items():
                if len(i) == 0:
                    continue
                if type(i[0]) in [str, np.str_]:
                    newDataset.partitions[name] = list(filter(lambda x: x in newDataset.files, i))
                else:
                    valid_ids = np.intersect1d(i, idx)
                    newDataset.partitions[name] = [np.where(idx == valid_ids[i])[0] for i in range(len(valid_ids))]
                    if len(newDataset.partitions[name]) > 0:
                        newDataset.partitions[name] = np.concatenate(newDataset.partitions[name])
        return newDataset

    def retrieve_from_class_id(self, meta_id, task, ids=None, exclusive=False):
        """
        retrieves the sub-dataset containing the targeted classes ƒor a given task.
        :param meta_id: list of ids to filter
        :type meta_id: int, list(int), np.ndarray
        :param str task: task to filter
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        :return: a filtered sub-dataset
        :rtype: :py:class:`Dataset`
        """
        ids = self.get_ids_from_class(meta_id, task, exclusive=exclusive, ids=ids)
        return self.retrieve(list(ids))


    def retrieve_from_class(self, meta_class, task, ids=None, exclusive=False):
        """
        retrieves the sub-dataset containing the targeted classes ƒor a given task.
        :param meta_id: list of ids to filter
        :type meta_id: int, list(int), np.ndarray
        :param str task: task to filter
        :param ids: (optional) constrains the search to the provided ids
        :type ids: np.ndarray or list(int)
        :param bool exclusive: only select data ids with every provided ids, in case of multi-label information (default: False)
        :return: a filtered sub-dataset
        :rtype: :py:class:`Dataset`
        """
        meta_id = [self.classes[task][m] for m in checklist(meta_class)]
        return self.retrieve_from_class_id(meta_id, task, exclusive=exclusive, ids=ids)

    
    def get_options_dict(self):
        """
        retrieve the dataset parameters as a dictionary.
        :return: parameters dictionary.
        :rtype: dict
        """
        return {'dataPrefix':self.dataPrefix, 'dataDirectory':self.dataDirectory, 'analysisDirectory':self.analysisDirectory, 'metadataDirectory':self.metadataDirectory,
                'importCallback':self.importCallback, 'types':self.types, 'tasks':self.tasks, 'taskCallback':self.taskCallback, 'metadataFiles':self.metadataFiles,
                'verbose':self.verbose, 'forceUpdate':self.forceUpdate, 'checkIntegrity':self.checkIntegrity}


    def save(self, location=None, **add_args):
        """
        save the dataset as a pickle. can be loaded using the corresponding class using the class method :py:func:Dataset.load
        :param str location: target path
        :param dict add_args: additional keys to save with the dataset.
        """
        save_dict = {'data':self.data,
                     'metadata':self.metadata,
                     'files':self.files,
                     'hash':self.hash,
                     'labels':self.labels,
                     'classes':self.classes,
                     'options_dict':self.get_options_dict(),
                     **add_args}
        location = location or self.analysisDirectory + '/' + self['transformName'] + '/dataset_pickle.npz'
        location = os.path.splitext(location)[0] + '.vs'
        with open(location, 'wb') as f:
            dill.dump({'dataset':save_dict, **add_args}, f)

    @classmethod
    def load(cls, location):
        """
        load a dataset saved using the :py:class:`Dataset` method.
        :param str location: path of loaded file
        """
        location = os.path.splitext(location)[0] + '.vs'
        with open(location, 'rb') as f:
            loaded = dill.load(f)
        dataset = cls(loaded['dataset']['options_dict'])
        dataset.data = loaded['dataset']['data']; dataset.metadata = loaded['dataset']['metadata']
        dataset.files = loaded['dataset']['files']; dataset.hash = loaded['dataset']['hash']
        dataset.labels = loaded['dataset']['labels']; dataset.classes = loaded['dataset']['classes']
        del loaded['dataset']
        return dataset, loaded

    """
    ###################################
    
    Listing functions
    
    ###################################
    """

    def list_directory(self, check=True):
        """
        Fill the list of files from a direct recursive path of a folder.
        This folder is specified at object construction with the 
        dataDirectory attribute

        :param bool check: removes the file if the corresponding transform is not found (default: False)
        """
        # The final set of files
        filesList = []

        # Use glob to find all recursively
        for dirpath,_,filenames in os.walk(self.dataDirectory):
            for f in filenames:
                if f.endswith(tuple(self.types)):
                    filesList.append(os.path.abspath(os.path.join(dirpath, f)))
                                        
        hashList = {}
        curFile = 0;
        
        # Parse the list to have a hash
        for files in filesList:
            hashList[files] = curFile
            curFile = curFile + 1
            
        # Print info if verbose mode
        # Save the lists
        self.files = filesList;
        self.hash = hashList;
        if (self.verbose):
            print('[Dataset][List directory] Found ' + str(len(self.files)) + ' files.');
    
    def list_input_file(self, inputFile=None, check=False):
        """ 
        List informations over a dataset (based on a given inputs.txt file).
        Will fill the files atributte of the instance
        :param str inputFile: inputs.txt to load (default : metadata/inputs.txt)
        :param bool check: checks the input files (default: False)
        """
        inputFile = inputFile or self.metadataDirectory + '/inputs.txt'
        # Try to open the given file
        if check:
            fileCheck = open(inputFile, "r")
            if (fileCheck is None):
                print('[Dataset][List file] Error - ' + inputFile + ' does not exists !.')
                return None
        # Create data structures
        filesList = []
        hashList = {}
        curFile = 0
        testFileID = None
        for line in fileCheck:
            if line[0] != "#":
                vals = re.search("^([^\t]+)\t?(.+)?$", line)
                audioPath = vals.group(1)
                if (audioPath is None): 
                    audioPath = line
                if (self.checkIntegrity):
                    testFileID = open(self.dataPrefix + '/' + audioPath, 'r')
                if (self.checkIntegrity) and (testFileID is None):
                    if (self.verbose): 
                        print('[Dataset][List file] Warning loading ' + inputFile + ' - File ' + self.dataPrefix + audioPath + ' does not exists ! (Removed from list)')
                else:
                    if (testFileID):
                        testFileID.close()
                    if (self.hash.get(self.dataPrefix + '/' + audioPath) is None):
                        filesList.append(self.dataPrefix + '/' + audioPath)
                        hashList[filesList[curFile]] = curFile
                        curFile = curFile + 1
        fileCheck.close()
        # Save the lists
        self.files = filesList
        self.hash = hashList
        

    """
    ###################################
    
    # Metadata loading functions
    
    ###################################
    """

    def import_metadata(self, fileName, task, callback):
        """  
        Import the metadata given in a file, for a specific task
        The function callback defines how the metadata should be imported
        All these callbacks should be in the importUtils file 
        
        Parameters
        ----------
        :param str fileName: Path to the file containing metadata
        :param str task: name of the metadata task
        :param function callback:  Callback defining how to import metadata

        """
        # Try to open the given file
        try:
            fileCheck = open(fileName, "r")
        except:
            print('[Dataset][List file] Error - ' + str(fileName) + ' does not exists !.')
            return None
        # Create data structures
        metaList = [None] * len(self.files)
        curFile, curHash = len(self.files), -1
        testFileID = None
        classList = {"_length":0}
        for line in fileCheck:
            line = line[:-1]
            if line[0] != "#" and len(line) > 1:
                vals = line.split('\t') #re.search("^(.+)\t(.+)$", line)
                if len(vals) != 2:
                   continue
                audioPath, metaPath = vals[0], (vals[1] or "") #vals.group(1), vals.group(2)
                if (audioPath is not None):
                    fFileName = self.dataPrefix + '/' + audioPath;
#                    fAnalysisName = 
                    if (self.checkIntegrity):
                        try:
                            testFileID = open(fFileName, 'r')
                            assert(fFileName in self.files)
                        except:
                            if self.verbose:
                                print('[Dataset][Metadata import] Warning loading task ' + task + ' - File ' + fFileName + ' does not exists ! (Removed from list)')
                            continue

                    if (testFileID):
                        testFileID.close()
                    # if (self.hash.get(fFileName) is None):
                    #     self.files.append(fFileName)
                    #     self.hash[self.files[curFile]] = curFile
                    #     curHash = curFile;
                    #     curFile = curFile + 1
                    # else:
                    #@pdb.set_trace()
                    if fFileName in self.hash.keys():
                        curHash = self.hash[fFileName];
                        if (len(metaList) - 1 < max(checklist(curHash))):
                            metaList.append("");
                        curHash = checklist(curHash)
                        for ch in curHash:
                            metaList[ch], classList = callback(metaPath, classList, {"prefix":self.dataPrefix, "files":self.files})
                    else:
                        pass
                else:
                    metaList.append({})

        # Save the lists
        self.metadata[task] = np.array(metaList);
        label_files = '/'.join(fileName.split('/')[:-1])+'/classes.txt'
        if os.path.isfile(label_files):
            type_metadata = type(self.metadata[task][0])
            classes_raw = open(label_files, 'r').read().split('\n')
            classes_raw = [tuple(c.split('\t')) for c in classes_raw]
            classes_raw = list(filter(lambda x: len(x)==2, classes_raw))
            self.classes[task] = {v:type_metadata(k) for k,v in classes_raw}
        else:
            if classList is not None:
                self.classes[task] = classList;
            else:
                self.classes[task] = {k:k for k in set(self.metadata[task])}

    def import_metadata_tasks(self, sort=True):
        """
        imports in the :py:class:`Dataset` object the metadata corresponding to the recorded tasks.
        :param bool sort: is metadata sorted (default : True)
        """
        if len(self.taskCallback) != len(self.tasks):
            self.taskCallback = [None]*len(self.tasks) if len(self.taskCallback) == 0 else self.taskCallback
            for t in range(len(self.tasks)):
                self.taskCallback[t] = self.retrieve_callback_from_path(self.metadataDirectory, self.tasks[t]) or metadataCallbacks["default"] or []
        if len(self.metadataFiles) == 0:
            self.metadataFiles = [None]*len(self.tasks) if len(self.metadataFiles) == 0 else self.metadataFiles
            for t in range(len(self.tasks)):
                self.metadataFiles[t] = self.metadataDirectory + '/' + self.tasks[t] + '/metadata.txt' or self.metadataDirectory + '/metadata.txt'
        for t in range(len(self.tasks)):
            self.import_metadata(self.metadataFiles[t], self.tasks[t], self.taskCallback[t])
        if sort:
            for t in self.tasks:
                self.sort_metadata(t)

    def sort_metadata(self, task):
        """
        sort the given classes of the metadata *task*, such that class_ids are affected to the class names with a given
        order
        :param str task: sorted task
        """
        private_keys = ['_length', None]
        private_dict = {p: self.classes[task].get(p) for p in private_keys}
        class_names = list(filter(lambda x: x not in private_keys, self.classes[task].keys()))
        clear_class_dict = {k: self.classes[task][k] for k in class_names}
        try:
            class_names = {float(k):v for k, v in clear_class_dict.items()}
            class_keys = list(class_names.keys())
            sorted_classes = np.argsort(np.array(class_keys))
            inverse_class_hash = {v:k for k, v in class_names.items()}
            for i in range(len(self.metadata[task])):
                original_class = inverse_class_hash[self.metadata[task][i]]
                class_idx = class_keys.index(original_class)
                self.metadata[task][i] = np.where(sorted_classes == class_idx)[0]

            new_classes = {c: i for i, c in enumerate(np.array(class_keys)[sorted_classes])}
            self.classes[task] = {**new_classes, **private_dict}
        except Exception as e:
            print('Task %s does not seem sortable.'%task)
            print(e)
            pass

    def retrieve_tasks(self):
        """
        automatically retrieves tasks recorded in the given dataset.
        """
        tasks = []
        folders = list(filter(lambda x: os.path.isdir(self.metadataDirectory+'/'+x), os.listdir(self.metadataDirectory)))
        for f in folders:
            if os.path.isfile("%s/%s/metadata.txt"%(self.metadataDirectory, f)):
                tasks.append(f)
        self.tasks = tasks


    """
    ###################################
    
    # Data import functions
    
    ###################################
    """
    
    def import_data(self):
        """
        data import callback. has be to overriden by specific sub-classes.
        """
        print('[Dataset][Import] Warning, undefined function in generic dataset.')
        self.data = []


    def retrieve_callback_from_path(self, metadataDirectory, task):
        """
        retrieve the appropriate callback for the given task.
        :param str metadataDirectory: metadata directory
        :param str task: target task
        :return:
        """
        try:
            with open("%s/%s/callback.txt"%(metadataDirectory, task)) as f:
                return getattr(data_metadata, re.sub('\n', '', f.read()))
        except FileNotFoundError:
            pass
        return

    """
    ###################################
    
    # Partitioning functions
    
    ###################################
    """

    def construct_partition(self, tasks, partitionNames, partitionPercent,  balancedClass=True, equalClass=False):
        """
        Construct a random/balanced partition set for each dataset
        Only takes indices with valid metadatas for every task
        now we can only balance one task

        :param tasks: balanced tasks
        :type tasks: str, list(str)
        :param list partitionNames: names of partition (list of str)
        :param list partitionPercent: relative proportion of each partition (summing up to 1)
        :param bool balancedClass: has the partition to be balanced for each class across partitions
        :param bool equalClass: enforces each class to be present with the same number of instances
        """
        #TODO balancing broken
        if (type(tasks) is str):
            tasks = [tasks]
        if len(tasks)==0 and balancedClass:
            tasks = self.tasks
        if (balancedClass is True):
            balancedClass = tasks[0]
        # Checking if tasks exist
        for t in tasks:
            if (self.metadata[t] is None): 
                print("[Dataset] error creating partitions : " + t + " does not seem to exist")
                return None
        # making temporary index from mutal information between tasks
        mutual_ids = [];

        data_shape = len(self.data)
        for i in range(data_shape):
            b = True
            for t in tasks: 
                b = b and (self.metadata[t][i] is not None)
            if (b):
                mutual_ids.append(i)
        # Number of instances to extract
        nbInstances = len(mutual_ids)
        if (len(mutual_ids) == 0):
            if type(self.metadata[tasks[1]]) is np.ndarray:
                nbInstances = (self.metadata[tasks[0]].shape[0])
            else: 
                nbInstances = len(self.metadata[tasks[0]])
            for i in range(nbInstances):
                mutual_ids[i] = i
        partitions = {}
        runningSum = 0;
        partSizes = np.zeros(len(partitionNames))
        for p in range(len(partitionNames)):
            partitions[partitionNames[p]] = [];
            if (p != len(partitionNames)):
                partSizes[p] = np.floor(nbInstances * partitionPercent[p]);
                runningSum = runningSum + partSizes[p];
            else:
                partSizes[p] = nbInstances - runningSum;
        # Class-balanced version
        if balancedClass:
            # Perform class balancing
            curMetadata = self.metadata[balancedClass];
            curClasses = self.classes[balancedClass];
            nbClasses = curClasses["_length"];
            countclasses = np.zeros(nbClasses);
            classIDs = {};
            # Count the occurences of each class
            for idC in range(len(mutual_ids)):
                s = mutual_ids[idC]
                countclasses[curMetadata[s]] = countclasses[curMetadata[s]] + 1;
                # Record the corresponding IDs
                if (not classIDs.get(curMetadata[s])):
                    classIDs[curMetadata[s]] = [];
                classIDs[curMetadata[s]].append(s);
            if equalClass:
                minCount = np.min(countclasses) 
                for c in range(nbClasses):
                    countclasses[c] = int(minCount);
            for c in range(nbClasses):
                if (classIDs[c] is not None):
                    curIDs = np.array(classIDs[c]);
                    classNb, curNb = 0, 0;
                    shuffle = np.random.permutation(int(countclasses[c]))
                    for p in range(len(partitionNames)):
                        if equalClass:
                            classNb = np.floor(partSizes[p] / nbClasses); 
                        else:
                            classNb = np.floor(countclasses[c] * partitionPercent[p])
                        if (classNb > 0):
                            for i in range(int(curNb), int(curNb + classNb - 1)):
                                partitions[partitionNames[p]].append(curIDs[shuffle[np.min([i, shuffle.shape[0]])]])
                            curNb = curNb + classNb;
        else:
            # Shuffle order of the set
            shuffle = np.random.permutation(len(mutual_ids))
            curNb = 0
            for p in range(len(partitionNames)):
                part = shuffle[int(curNb):int(curNb+partSizes[p]-1)]
                for i in range(part.shape[0]):
                    partitions[partitionNames[p]].append(mutual_ids[part[i]])
                curNb = curNb + partSizes[p];
        for p in range(len(partitionNames)):
            self.partitions[partitionNames[p]] = np.array(partitions[partitionNames[p]]);
        return partitions

    def construct_partition_from_files(self, tasks, partitionNames, partitionPercent, balancedClass=True):
        """
        Construct partitions, such that indices corresponding the same files are gathered in the same partition.

        Parameters
        ----------
        partitionNames : list of str
            List of the names to be added to the ``partitions`` attribute
        partitionFiles : list of str
            List of files from which to import partitions

        """
        files = list(set(self.files))
        full_idxs = np.random.permutation(len(files))
        splits_idxs = []
        split_lens = [int(len(full_idxs) * p) for p in partitionPercent]
        idxs_tmp = full_idxs
        for s in split_lens:
            splits_idxs.append(idxs_tmp[:s])
            idxs_tmp = idxs_tmp[s:]
        partitions = {}; partitions_files = {}
        for i,name in enumerate(partitionNames):
            partitions[name] = sum([self.hash[files[f]] for f in splits_idxs[i]], [])
            partitions_files[name] = np.array(files)[splits_idxs[i]]
        self.partitions = partitions_files
        return partitions


    def construct_partition_files(self, partitionNames, partitionFiles):
        """ 
        Constructing partitions from a given set of files.
        Each of the partition file given should contain a list of files that
        are present in the original dataset list
        
        Parameters
        ----------
        partitionNames : list of str
            List of the names to be added to the ``partitions`` attribute
        partitionFiles : list of str
            List of files from which to import partitions
        
        """
        def findFilesIDMatch(fileN):
            fIDRaw = open(fileN, 'r');
            finalIDx = []
            if (fIDRaw is None):
                print('  * Annotation file ' + fileN + ' not found.');
                return None
            # Read the raw version
            for line in fIDRaw:
                data = line.split("\t")
                pathV, fileName = path.split(data[0])
                fileName, fileExt = path.splitext(fileName)
                for f in range(len(self.files)):
                    path2, fileName2 = path.split(self.files[f])
                    fileName2, fileExt2 = path.splitext(fileName2)
                    if (fileName == fileName2):
                        finalIDx.append(f)
                        break;
            return np.array(finalIDx);
        for p in range(len(partitionNames)):
            self.partitions[partitionNames[p]] = findFilesIDMatch(partitionFiles[p]);
        return self.partitions;


    """
    ###################################
    #
    # Data indexing functions
    #
    ###################################
    """

    def create_batches(self, batchSize, partition=None, balancedTask=None):
        """
        divide the data in a set of batches.
        :param batchSize: size of batches
        :param str partition: selected partition (default: None)
        :param str balancedTask: balance batches across a given task (default: None)
        :return: batch indices
        :rtype: np.ndarray
        """
        """
        Create a set of batches balanced in terms of classes
        modif axel 31/04/17 : choix de la partition
        (for non balanced batches just let task to nil)
        (to fetch the whole dataset let partition to nil)
        (#TODO what if I could put myself to nil
        """
        finalIDs = {};
        if balancedTask:
            if balancedTask == True:
                balancedTask = self.task[1]
            if (self.metadata[balancedTask] is None):
                print('[Dataset] Error creating batch : ' + balancedTask + ' does not seem to exist') 
                return None
            if partition:
                partition_ids = self.partitions[partition] 
            else:
                partition_ids = range(1, len(self.metadata[balancedTask]))
            labels = np.array(self.metadata[balancedTask])[partition_ids]
            nbClasses = self.classes[balancedTask]["_length"]
            countLabels = np.zeros(nbClasses)
            classIDs = {};
            # Count the occurences of each class
            for s in range(partition_ids.shape[0]):
                countLabels[labels[s]] = countLabels[labels[s]] + 1;
                # Record the corresponding IDs
                if (classIDs.get(labels[s]) is None):
                    classIDs[labels[s]] = [];
                classIDs[labels[s]].append(partition_ids[s])
            minClassNb = np.min(countLabels)
            finalIDs = np.zeros(int(minClassNb * nbClasses));
            # Then perform a randperm of each class
            for c in range(nbClasses):
                curIDs = np.array(classIDs[c])
                curIDs = curIDs[np.random.permutation(curIDs.shape[0])]
                setPrep = (np.linspace(0, ((minClassNb - 1) * nbClasses), minClassNb) + (c - 1)).astype(int)
                finalIDs[setPrep] = curIDs[:int(minClassNb)]
            # Return class-balanced IDs split by batch sizes
            overSplit = finalIDs.shape[0] % batchSize
            finalIDs = np.split(finalIDs[:-overSplit], finalIDs[:-overSplit].shape[0] / batchSize);
        else:
            if partition:
                partition_ids = self.partitions[partition] 
            else:
                partition_ids = range(1, len(self.data))
            indices = np.random.permutation(partition_ids.shape[0])
            curIDs = partition_ids[indices]
            overSplit = curIDs.shape[0] % batchSize
            finalIDs = np.split(curIDs[:-overSplit], curIDs[:-overSplit].shape[0] / batchSize)
        # Remove the last if it is smaller
        # if finalIDs[#finalIDs]:size() < batchSize then finalIDs[#finalIDs] = nil; end
        return finalIDs;

    def flatten_data(self, selector=lambda x: x, window=None, window_overlap=0.5, merge_mode="min", stack=False):
        """
        if the data is built of nested arrays, flattens the data to be a single array. Typically, if each item of data
        is a sub-array of size (nxm), flatten_data concatenates among the first axis, and can optionally window among
        the second.
        If the second dimension of each sub-arrays do not match, it can be cropped (*merge_mode* min)
        or padded (*merge_mode* max)

        :param function selector: a lambda selector, that picks the wanted data in each sub-array
        :param int window: size of the window (default: None)
        :param float window_overlap: overlapping of windows
        :param str merge_mode: merging mode of nested array, if the dimensions are not matching ("min" or "max").
        """

        def window_data(chunk, window, window_overlap):
            dim = 0
            n_windows = (chunk.shape[dim] - window) // int(window * window_overlap)
            chunk_list = []
            if n_windows >= 0:
                for i in range(n_windows+1):
                    chunk_list.append(np.expand_dims(np.take(chunk, range(int(i*window_overlap), int(i*window_overlap) + window), axis=dim),dim))
            else:
                pads = [(0,0)]*len(chunk.shape)
                pads[dim] = (0, window - chunk.shape[dim])
                chunk_list = [np.pad(chunk, pads, mode="edge")]
            if len(chunk_list) > 2:
                return np.concatenate(chunk_list, axis=dim)
            else:
                return chunk_list[0]

        assert merge_mode in ['min', 'max']

        dataBuffer = []
        newMetadata = {}
        for k, v in self.metadata.items():
            newMetadata[k] = []
        newFiles = []
        revHash = {}

        # new hash from scratch
        newHash = dict(self.hash)
        for k, v in self.hash.items():
            newHash[k] = []

        # filter dataset
        running_sum = 0
        min_size = None; max_size = None
        for i in range(len(self.data)):
            chunk_to_add = selector(self.data[i])
            if window is not None:
                chunk_to_add = window_data(chunk_to_add, window, window_overlap)
            if chunk_to_add.ndim == 1:
                chunk_to_add = np.reshape(chunk_to_add, (1, chunk_to_add.shape[0]))
            # update minimum content shape
            if min_size is None:
                min_size = chunk_to_add.shape[1:] if chunk_to_add.ndim > 2 else (chunk_to_add.shape[1],)
                max_size = chunk_to_add.shape[1:] if chunk_to_add.ndim > 2 else (chunk_to_add.shape[1],)
            else:
                current_min_size = chunk_to_add.shape[1:] if chunk_to_add.ndim > 2 else (chunk_to_add.shape[1],)
                min_size = np.minimum(min_size, current_min_size); max_size = np.maximum(max_size, current_min_size)
            if stack:
                chunk_to_add = chunk_to_add[np.newaxis]
            dataBuffer.append(chunk_to_add)
            for k, _ in newMetadata.items():
                newMetadata[k].extend([self.metadata[k][i]]*dataBuffer[i].shape[0])
            if len(self.files) > 0:
                newFiles.extend([self.files[i]]*dataBuffer[i].shape[0])
            running_sum += dataBuffer[i].shape[0]

        target_size = min_size if merge_mode == "min" else max_size
        newData = np.zeros((running_sum, *target_size), dtype=self.data[0].dtype)

        running_id = 0
        for i in range(len(dataBuffer)):
            newData[running_id:(running_id+dataBuffer[i].shape[0]), :] = data_utils.crop_or_pad(dataBuffer[i], target_size)
            if len(newFiles) > 0:
                newHash[self.files[i]].extend(range(running_id, running_id+dataBuffer[i].shape[0]))
            for idx in range(running_id, running_id+dataBuffer[i].shape[0]):
                revHash[idx] = i

            running_id+=dataBuffer[i].shape[0]
            

        self.data = newData
        self.metadata = newMetadata
        for k,v in newMetadata.items():
            newMetadata[k] = np.array(v)
        self.files = newFiles
        self.hash = newHash
        self.revHash = revHash


    def window(self, window_size, window_overlap, axis=1):
        """
        window the data array among the given dimension.
        :param window_size: size of the window
        :param window_overlap: overlap of the window
        :param axis: window axis
        :return:
        """

        if issubclass(type(self.data), list):
            self.data = [data_utils.window_data(self.data[i], window_size, window_overlap, axis) for i in range(len(self.data))]
        else:
            self.data = data_utils.window_data(self.data, window_size, window_overlap)


    def return_padded(self, *args, max_len=None):
        if 0 in args:
            raise Exception('cannot pad dimension 0')
        if max_len is None:
            max_len = tuple([None]*len(args))

        self.padded_dims = set(args).union(self.padded_dims)
        for i,dim in enumerate(args):
            self.padded_lengths[dim] = max([d.shape[dim-1] for d in self.data]) if max_len[i] is None else max_len[i]



    def remove_files(self, n_files, shuffle=True):
        """
        just keep a given number of files and drop the rest
        :param n_files: number of kept audio files
        :param shuffle: randomize audio files (default: True)
        """

        assert n_files < len(self.files), "number of amputated files greater than actual number of files!"
        if shuffle:
            ids = np.random.permutation(len(self.files))[:n_files]
        else:
            ids = np.arange(n_files)
        self.files = [self.files[i] for i in ids]
        self.hash = {self.files[i]:i for i in range(len(self.files))}
        for k, v in self.partitions.items():
            if type(v[0]) == str:
                self.partitions[k] = list(filter(lambda x: x in self.files, v))


    def translate_files(self, files):
        """
        translate an incoming list of data paths with the current root directory.
        :param list files: list of file paths to translate
        :return: translated files
        :rtype: list(str)
        """

        if self.dataPrefix.split('/')[-1] not in files[0]:
            raise Exception('given files do not seem in current dataset')
        oldRoot = files[0][:re.search(self.dataPrefix.split('/')[-1], files[0]).regs[0][1]]
        translated_files = list({re.sub(oldRoot, self.dataPrefix, k) for k in files})
        translated_files = list(filter(lambda x: x in self.hash.keys(), translated_files))
        return translated_files


    def filter_files(self, files):
        """
        returns a sub-dataset containing data only extracted from the given array of files
        :param list files: list of files to retrieve
        :returns: sub-dataset
        :rtype: :py:class:`Dataset`
        """
        files = list(set(files))
        translated_files = self.translate_files(files)
        ids = list(set(sum([checklist(self.hash[k]) for k in translated_files], [])))
        return self.retrieve(ids)

    def filter_classes(self, class_names, task):
        class_names = checklist(class_names)
        ids = set(); 
        for cn in class_names:
            ids = ids.union(set(self.get_ids_from_class(self.classes[task][cn], task)))
        valid_ids = list(set(list(range(len(self)))).difference(ids))
        return self.retrieve(valid_ids)


    """
    ###################################
    #
    # Conversion / extraction functions
    #
    ###################################
    """

    def get_meta_dataset(self, tasks):
        """
        Extracts a :py:class:`Dataset` object whose data is the metadata of the current dataset.
        :param list tasks: tasks extracted
        :returns: sub-dataset
        :rtype: :py:class:`Dataset`
        """
        if issubclass(type(tasks), (list, tuple)):
            assert None not in [t in self.metadata.keys() for t in tasks]
            meta_dataset = copy.deepcopy(self)
            meta_dataset.data = [meta_dataset.metadata[t] for t in tasks]
        else:
            meta_dataset = copy.deepcopy(self)
            meta_dataset.data = meta_dataset.metdata[tasks]
        return meta_dataset



def dataset_from_torch(name, transform=None, target_transform=None):
    """
    Creates a :py:class:`Dataset` from a torchvision dataset.
    :param str name: name of torchvision module
    :param transform: transforms
    :param target_transform: target_transforms
    :return: torch dataset
    :rtype: :py:class:`Dataset`
    """
    current_path = os.path.dirname(__file__)+'/toys'
    train_dataset = getattr(torchvision.datasets, name)(current_path, transform=transform, target_transform=target_transform, download=True, train=True)
    test_dataset = getattr(torchvision.datasets, name)(current_path, transform=transform, target_transform=target_transform, download=True, train=False)
    full_dataset = Dataset({'dataPrefix':current_path})
    full_dataset.data = np.concatenate([train_dataset.train_data.float().numpy(), test_dataset.test_data.float().numpy()], axis=0)
    full_dataset.partitions = {'train':np.arange(train_dataset.train_data.shape[0]),
                               'test':(train_dataset.train_data.shape[0] + np.arange(test_dataset.test_data.shape[0]))}
    #pdb.set_trace()
    full_dataset.metadata = {'class':np.concatenate([train_dataset.train_labels.numpy(), test_dataset.test_labels.numpy()])}
    full_dataset.classes = {'class':{k:k for k in range(full_dataset.metadata['class'].min(), full_dataset.metadata['class'].max()+1)}}
    full_dataset.tasks = ['class']
    return full_dataset


