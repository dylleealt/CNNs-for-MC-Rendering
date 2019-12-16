import utils as *

def load_and_preprocess_exr(input_file, gt_file):
	data = read_exr(input_file)
	gt = read_exr(gt_file)

	# clip specular component to avoid negative logs
	data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
  	data['specularVariance'] = np.clip(data['specularVariance'], 0, np.max(data['specularVariance']))
  	gt['specular'] = np.clip(gt['specular'], 0, np.max(gt['specular']))
  	gt['specularVariance'] = np.clip(gt['specularVariance'], 0, np.max(gt['specularVariance']))

	# save reference data to calculate error
	data['Reference'] = np.concatenate((diff_ref[:,:,:3].copy(), spec_ref[:,:,:3].copy()), axis=2)
	data['FinalGt'] = gt['default']

	# preprocess diffuse and specular components
	data['diffuse'] = preprocess_diffuse(data['diffuse'], data['albedo'])
	data['diffuseVariance'] = preprocess_diff_var(data['diffuseVariance'], data['albedo'])
	data['specular'] = preprocess_specular(data['specular'])
	data['specularVariance'] = preprocess_spec_var(data['specularVariance'], data['specular'])

	# preprocess depth
	max_depth = np.max(data['depth'])
	data['depth'] = np.clip(data['depth'], 0, max_depth)
	# normalize depth
	if (max_depth != 0):
		data['depth'] /= max_depth
		data['depthVariance'] /= max_depth * max_depth

	# calculate gradients of features (not including variances)
	data['gradNormal'] = gradients(data['normal'][:, :, :3].copy())
	data['gradDepth'] = gradients(data['depth'][:, :, :1].copy())
	data['gradAlbedo'] = gradients(data['albedo'][:, :, :3].copy())
	data['gradSpecular'] = gradients(data['specular'][:, :, :3].copy())
	data['gradDiffuse'] = gradients(data['diffuse'][:, :, :3].copy())
	data['gradIrrad'] = gradients(data['default'][:, :, :3].copy())

	# concatenate variances and gradients to each features
	data['diffuse'] = np.concatenate((data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
	data['specular'] = np.concatenate((data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)
	data['normal'] = np.concatenate((data['normalVariance'], data['gradNormal']), axis=2)
	data['depth'] = np.concatenate((data['depthVariance'], data['gradDepth']), axis=2)

	# construct final diffuse and specular components
	x_diff = np.concatenate((data['diffuse'],
							data['normal'],
							data['depth'],
							data['gradAlbedo']), axis=2)

	x_spec = np.concatenate((data['specular'],
							data['normal'],
							data['depth'],
							data['gradAlbedo']), axis=2)

	data['Diffuse'] = x_diff
  	data['Specular'] = x_spec

	# remove unwanted channels
	channels_to_delete = []
	for k, v in data.items():
		if not k in ['Diffuse, Specular, Reference, FinalGt']:
			channels_to_delete.append(k)
	for k in channels_to_delete:
    	del data[k]

	return data

def load_dataset(filepath, batch_size, shuffle_buffer_size=250000, n_threads=2, is_testing=False):
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
	if is_testing:
		if filepath.endswith('.exr'):
			return load_and_preprocess_exr(filepath, None)
		else:
			dir_path = filepath + '/*.exr'
    		dataset = tf.data.Dataset.list_files(dir_path)
			dataset = dataset.map(map_func=load_and_preprocess_exr, num_parallel_calls=n_threads)
	else:
		dir_path = filepath + '/*.exr'
		dataset = tf.data.Dataset.list_files(dir_path)
    	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
		dataset = dataset.map(map_func=load_and_preprocess_exr, num_parallel_calls=n_threads)
		# crop and prune patches
		dataset = dataset.batch(batch_size, drop_remainder=True)
		dataset = dataset.prefetch(1)
		
	return dataset
