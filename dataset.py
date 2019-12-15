import utils

def load_and_preprocess_exr(filepath):
	pass

def load_dataset(filepath, batch_size, shuffle_buffer_size=250000, n_threads=2):
	"""
    Given a directory and a batch size, the following method returns a dataset iterator that can be queried for 
    a batch of images

    :param dir_name: a batch of images
    :param batch_size: the batch size of images that will be trained on each time
    :param shuffle_buffer_size: representing the number of elements from this dataset from which the new dataset will 
    sample
    :param n_thread: the number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
	if filepath.endswith('.exr'):
		return load_and_preprocess_exr(filepath)
	else:
		dir_path = filepath + '/*.exr'
    	dataset = tf.data.Dataset.list_files(dir_path)

    	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
		dataset = dataset.map(map_func=load_and_preprocess_exr, num_parallel_calls=n_threads)
		dataset = dataset.batch(batch_size, drop_remainder=True)
		dataset = dataset.prefetch(1)
		
	return dataset
