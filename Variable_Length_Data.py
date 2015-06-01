import numpy
import h5py
from fuel.datasets.hdf5 import H5PYDataset

def CreateHDF5():
    sizes = numpy.random.randint(3,9, size=(100,))
    train_image_features = [
            numpy.random.randint(256, size=(3, size, size)).astype('uint8')
            for size in sizes[:90]]
    test_image_features = [
            numpy.random.randint(256, size=(3, size, size)).astype('uint8')
            for size in sizes[90:]]

    train_vector_features = numpy.random.normal(size=(90,10)).astype('float32')
    test_vector_features = numpy.random.normal(size=(10,10)).astype('float32')
    train_targets = numpy.random.randint(10, size=(90,1)).astype('uint8')
    test_targets = numpy.random.randint(10, size=(10,1)).astype('uint8')

    f = h5py.File('dataset.hdf5', mode='w')
    vector_features = f.create_dataset(
         'vector_features', (100, 10), dtype='float32')
    targets = f.create_dataset(
         'targets', (100, 1), dtype='uint8')

    vector_features[...] = numpy.vstack(
         [train_vector_features, test_vector_features])
    targets[...] = numpy.vstack([train_targets, test_targets])


    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    all_image_features = train_image_features + test_image_features
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    image_features = f.create_dataset('image_features', (100,), dtype=dtype)
    image_features[...] = [image.flatten() for image in all_image_features]
    image_features.dims[0].label='batch'

    image_features_shapes = f.create_dataset(
         'image_features_shapes', (100, 3), dtype='int32')
    image_features_shapes[...] = numpy.array(
         [image.shape for image in all_image_features])
    image_features.dims.create_scale(image_features_shapes, 'shapes')
    image_features.dims[0].attach_scale(image_features_shapes)

    image_features_shape_labels = f.create_dataset(
         'image_features_shape_labels', (3,), dtype='S7')
    image_features_shape_labels[...] = [
         'channel'.encode('utf8'), 'height'.encode('utf8'),
         'width'.encode('utf8')]
    image_features.dims.create_scale(
         image_features_shape_labels, 'shape_labels')
    image_features.dims[0].attach_scale(image_features_shape_labels)

    split_dict = {
         'train': {'vector_features': (0, 90), 'image_features': (0, 90),
                   'targets': (0, 90)},
         'test': {'vector_features': (90, 100), 'image_features': (90, 100),
                  'targets': (90, 100)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

def testHDF5():
    train_set = H5PYDataset('dataset.hdf5', which_set='train')
    test_set = H5PYDataset('dataset.hdf5', which_set='test')

    print(train_set.num_examples, test_set.num_examples)

    print(train_set.provides_sources)

    print(train_set.axis_labels['image_features'])

    print(train_set.axis_labels['vector_features'])

    print(train_set.axis_labels['targets'])

    handle = train_set.open()
    data = train_set.get_data(handle, slice(0,10))
    print((data[0].shape, data[1].shape, data[2].shape))

    train_set = H5PYDataset(
            'dataset.hdf5', which_set='train', sources=('image_features',))
    print(train_set.axis_labels['image_features'])

    images, = train_set.get_data(handle, slice(0,10))
    train_set.close(handle)
    print(images[0].shape, images[1].shape)
