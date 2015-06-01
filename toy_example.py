import numpy
numpy.save(
    'train_vector_features.npy',
    numpy.random.normal(size=(90, 10)).astype('float32'))
numpy.save(
    'test_vector_features.npy',
    numpy.random.normal(size=(10, 10)).astype('float32'))
numpy.save(
    'train_image_features.npy',
    numpy.random.randint(2, size=(90, 3, 5, 5)).astype('uint8'))
numpy.save(
    'test_image_features.npy',
    numpy.random.randint(2, size=(10, 3, 5, 5)).astype('uint8'))
numpy.save(
    'train_targets.npy',
    numpy.random.randint(10, size=(90, 1)).astype('uint8'))
numpy.save(
    'test_targets.npy',
    numpy.random.randint(10, size=(10, 1)).astype('uint8'))

train_vector_features = numpy.load('train_vector_features.npy')
test_vector_features = numpy.load('test_vector_features.npy')
train_image_features = numpy.load('train_image_features.npy')
test_image_features = numpy.load('test_image_features.npy')
train_targets = numpy.load('train_targets.npy')
test_targets = numpy.load('test_targets.npy')

import h5py
f = h5py.File('dataset.hdf5', mode='w')
vector_features = f.create_dataset(
     'vector_features', (100, 10), dtype='float32')
image_features = f.create_dataset(
     'image_features', (100, 3, 5, 5), dtype='uint8')
targets = f.create_dataset(
     'targets', (100, 1), dtype='uint8')

vector_features[...] = numpy.vstack(
     [train_vector_features, test_vector_features])
image_features[...] = numpy.vstack(
     [train_image_features, test_image_features])
targets[...] = numpy.vstack([train_targets, test_targets])

vector_features.dims[0].label = 'batch'
vector_features.dims[1].label = 'feature'
image_features.dims[0].label = 'batch'
image_features.dims[1].label = 'channel'
image_features.dims[2].label = 'height'
image_features.dims[3].label = 'width'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'index'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
     'train': {'vector_features': (0, 90), 'image_features': (0, 90),
               'targets': (0, 90)},
     'test': {'vector_features': (90, 100), 'image_features': (90, 100),
              'targets': (90, 100)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()
