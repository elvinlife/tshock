"""    
tshock:data
author:@cyz
time:since 2018/2/2
"""
import numpy as np
from .type import _make_tuple

class AbsFileDataSrc(object):
    def load(self):
        """
        load the data from file
        :return:dict of data in numpy format
        """
        raise NotImplementedError()
    def save(self):
        raise NotImplementedError()

class PickleDataSrc(AbsFileDataSrc):
    def __init__(self, filename):
        self._filename = filename
        self._data = None
    def load(self, encoding='latin1'):
        if self._data is None:
            import pickle
            file = open(self._filename, 'rb')
            self._data = pickle.load(file,encoding=encoding)
            for key in self._data.keys():
                self._data[key] = np.array(self._data[key])
            file.close()
        return self._data
    def save(self, **kwargs):
        pass

class NpzDataSrc(AbsFileDataSrc):
    def __init__(self, filename):
        self._filename = filename
        self._data = None

    def load(self, *keys):
        if self._data is None:
            self._data = {}
            nlf = np.lib.format
            npz = np.load(self._filename)
            file = npz.zip.fp
            for key in npz.files:
                filename = '{}.npy'.format(key)
                npz.zip.open(filename)
                version = nlf.read_magic(file)
                shape, fortran_order, dtype = nlf.read_array_header_1_0(file) if version == (1, 0) \
                    else nlf.read_array_header_2_0(file)
                self._data[key] = np.memmap(
                    file, dtype=dtype, mode='r',
                    shape=shape, order='F' if fortran_order else 'C',
                    offset=file.tell()
                )
        return {key: self._data[key] for key in keys}

    def save(self, **kargs):
        pass


class AbstractDataStream(object):
    def __init__(self):
        self._keys = None
        self._batch_size = None
        self._size = None
        pass

    def next_batch(self):
        """
        :param size:batch size, default is None, which means all the data
        :return:a batch of data: in np array type
        """
        raise NotImplementedError()

    @property
    def keys(self):
        """
        :return:the dataset' keys
        """
        return self._keys

    @property
    def batch_size(self):
        """
        :return:current batch size
        """
        return self._batch_size

    @property
    def size(self):
        """
        :return:the data stream entry number
        """
        return self._size

    @keys.setter
    def keys(self, keys):
        self._keys = _make_tuple(keys)


class FileDataStream(AbstractDataStream):
    def __init__(self,
                 data: dict,
                 keys = None,
                 ):
        """
        :param data: dict, store the data
        :param keys: tuple, store the keys for the stream to generate
        """
        super(FileDataStream, self).__init__()
        self._data = data
        self._offset = 0
        self._size = None
        self._keys = data.keys() if keys == None else keys
        for key in self._keys:
            if self._size == None:
                self._size = len(data[key])
            elif self._size != len(data[key]):
                raise ValueError('All data must have same size')

    def next_batch(self, size=None):
        """
        :param size: batch size
        :return: tuple, the data order follows self._keys, in numpy format
        if size = None, return all the data;
        if size > self._size, loop and concatenate the stream data until get @p:size number entry
        else return data[offset:offset+size]
        """
        if size == None:
            return tuple([self._data[key] for key in self._keys])
        batch = self._next_batch(size)
        real_size = len(batch[0])
        while real_size < size:
            batch_ = self._next_batch(size-real_size)
            batch = tuple([np.concatenate((data, data_), 0) for data, data_ in zip(batch, batch_)])
            real_size = len(batch[0])
        return batch

    def _next_batch(self, size):
        if self._offset == 0:
            self.shuffle()
        l = self._offset
        r = l + size
        self._offset = r if r < self._size else 0
        return tuple([self._data[key][l:r].copy() for key in self._keys])

    def shuffle(self, num=3):
        perm = np.arange(self._size)
        for _ in range(num):
            np.random.shuffle(perm)
        for key in self._data.keys():
            self._data[key] = self._data[key][perm]

class MongoDataStream(AbstractDataStream):
    def __init__(self,
                 host, auth_db_name, user, passwd,
                 db_name, coll_name, keys
                 ):
        super(MongoDataStream, self).__init__()