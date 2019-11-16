import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import os, glob, subprocess, shutil


from sklearn.decomposition import PCA



def plot_reconstructions(input_data, output_data, **kwargs):

	N_test = min(kwargs.get('N_test', 12), len(input_data))

	recon_indices = np.random.choice(len(input_data), N_test, replace=False)


	img_w = 10
	img_h = img_w*(2/N_test)

	plt.close('all')
	fig, axes = plt.subplots(2, N_test, figsize=(img_w, img_h))

	for i in range(N_test):

		axes[0][i].axis('off')
		axes[1][i].axis('off')
		axes[0][i].imshow(input_data[recon_indices[i]], cmap='gray')
		axes[1][i].imshow(output_data[recon_indices[i]], cmap='gray')

	plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01,  wspace=0.05, hspace=0.05)


	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'recon_default.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()


def plot_2D_latent_space(latent_data, **kwargs):

	plt.close('all')
	fig, axes = plt.subplots(1, 1, figsize=(8, 8))

	if kwargs.get('label_list', None) is None:
		scatter_plot = plt.scatter(*latent_data.T, alpha=0.4)
	else:
		ll = kwargs.get('label_list', None)
		if isinstance(ll, torch.Tensor):
			ll = ll.detach().numpy()

		unique_labels = list(set(ll))
		N_unique = len(unique_labels)
		#print(f'{N_unique} unique labels: {unique_labels}')

		cmap = plt.get_cmap('tab10')
		norm = mpl.colors.BoundaryNorm(np.arange(-0.5,N_unique), cmap.N)
		scatter_plot = plt.scatter(*latent_data.T, alpha=0.3, c=ll, cmap=cmap, norm=norm, s=20)

		plt.colorbar(scatter_plot, ticks=list(range(N_unique)))


	if kwargs.get('highlight_points', None) is not None:

		highlight_pts = kwargs.get('highlight_points', None)
		plt.scatter(*highlight_pts.T, alpha=0.9, fc='red', ec='black', s=10)



	#plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, )

	plt.xlabel('$z_1$', fontsize=14)
	plt.ylabel('$z_2$', fontsize=14)

	plt.tight_layout()

	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'latent_space.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()


def plot_transformation(recon_data, **kwargs):

	N_steps = recon_data.shape[0]

	img_w = 8
	img_h = img_w*(1/N_steps)

	plt.close('all')
	fig, axes = plt.subplots(1, N_steps, figsize=(img_w, img_h))

	for i in range(N_steps):

		axes[i].axis('off')
		axes[i].imshow(recon_data[i], cmap='gray')

	plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, wspace=0.05, hspace=0.05)


	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'transform.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()




def plot_losses(loss_dict, **kwargs):

	cols = [
		'dodgerblue',
		'tomato',
		'mediumseagreen',
		'orange',
		'orchid',
		'cyan',
	]

	key_list = list(loss_dict.keys())
	N_plots = len(key_list)

	N_cols = np.ceil(np.sqrt(N_plots)).astype(int)
	N_rows = np.ceil(N_plots/N_cols).astype(int)

	plt.close('all')
	fig, axes = plt.subplots(N_rows, N_cols, figsize=(8, 8))

	for i,ax in enumerate(axes.flatten()):

		if i >= N_plots:
			break

		k = key_list[i]

		data = loss_dict[k]['data']

		ax.plot(data, 'o', markersize=2, alpha=0.3, color=cols[i])
		smoothed = smooth_data(data)
		ax.plot(*smoothed, '-', color='black')

		ax.set_xlim(-0.02*len(data), len(data))
		r = max(np.ptp(smoothed[1]), 0.05*np.abs(smoothed[1][0]))
		ax.set_ylim(np.min(data), np.max(smoothed[1]) + 0.15*r)



		if loss_dict[k].get('xlabel', None) is None:
			ax.set_xlabel('Batch', fontsize=12)
		else:
			ax.set_xlabel(loss_dict[k].get('xlabel', None), fontsize=12)

		if loss_dict[k].get('ylabel', None) is None:
			ax.set_ylabel(k, fontsize=12)
		else:
			ax.set_ylabel(loss_dict[k].get('ylabel', None), fontsize=12)

		if loss_dict[k].get('yscale', None) is None:
			ax.set_yscale('log')
		else:
			ax.set_yscale(loss_dict[k].get('yscale', None))




	plt.tight_layout()

	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'losses.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()


def plot_image_grid(img_grid, **kwargs):

	'''
	Assumes you pass a list of lists of images. Assumes each of the sublists
	are the same size.


	'''

	N_rows = len(img_grid)
	N_cols = len(img_grid[0])

	print(f'Plotting a {N_rows} by {N_cols} grid...')

	feat_grid = kwargs.get('feat_grid', None)

	plt.close('all')
	fig, axes = plt.subplots(N_rows, N_cols, figsize=(8*N_cols/N_rows, 8))

	for i in range(N_rows):
		for j in range(N_cols):

			axes[i][j].axis('off')


			axes[i][j].imshow(img_grid[i][j], cmap='gray')

			if feat_grid is not None:
				feats = feat_grid[i][j]
				eye_L = feats['eye_L']
				eye_R = feats['eye_R']
				eyes_midpt = feats['eyes_midpt']
				nose = feats['nose']
				axes[i][j].plot(*eye_L, 'go')
				axes[i][j].plot(*eye_R, 'ro')
				axes[i][j].plot(*eyes_midpt, 'bo')
				axes[i][j].plot(*nose, 'yo')


	plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, wspace=0.0, hspace=0.0)

	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'img_grid.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()




def create_interp_gif(interp_data, output_dir, rel_fname, **kwargs):

	gif_pics_dir = os.path.join(output_dir, 'gif_pics')
	if os.path.exists(gif_pics_dir):
		shutil.rmtree(gif_pics_dir)

	os.mkdir(gif_pics_dir)


	N_steps = interp_data.shape[0]


	for i in range(N_steps):
		plt.close('all')
		plt.figure(figsize=(2, 2))

		plt.axis('off')
		plt.imshow(interp_data[i], cmap='gray')

		plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, )

		fname = os.path.join(gif_pics_dir, f'{i}.png')
		plt.savefig(fname)

	gif_name = os.path.join(output_dir, rel_fname)
	imgs_to_gif(gif_pics_dir, gif_name, length=N_steps*0.015)






def imgs_to_gif(imgs_dir, output_fname, **kwargs):

	ext = '.png'

	length = 1000.0*kwargs.get('length', 5.0) # passed in seconds, turned into ms

	file_list = glob.glob(os.path.join(imgs_dir, f'*{ext}')) # Get all the pngs in the current directory
	N_files = len(file_list)
	delay = length/N_files
	print(f'Making gif from {N_files} pics, using delay of {delay} for total length of {length:.3f}')
	#print(file_list)
	list.sort(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
	#print(file_list)
	assert len(file_list) < 300, 'Too many files ({}), will probably crash convert command.'.format(len(file_list))

	check_call_arglist = ['convert'] + ['-delay', str(delay)] + file_list + [output_fname]
	#print(check_call_arglist)
	print('Calling convert command to create gif...')
	subprocess.check_call(check_call_arglist)
	print('done.')
	return output_fname



def plot_losses_hist(input_data, output_data, **kwargs):


	print('Data shape: ', input_data.shape)


	mse = np.mean((input_data - output_data)**2, axis=1)

	print('MSE shape: ', mse.shape)

	plt.close('all')
	n, bins, _ = plt.hist(mse, log=True, facecolor='dodgerblue', edgecolor='black', bins=25)

	if kwargs.get('highlight_pts', None) is not None:

		highlight_pts = kwargs.get('highlight_pts', None)
		in_pts = highlight_pts['in_pts']
		out_pts = highlight_pts['out_pts']

		mse_pts = np.mean((in_pts - out_pts)**2, axis=1)

		print(mse_pts)

		for pt in mse_pts.tolist():
			plt.axvline(pt, linestyle='dashed', color='tomato')

	plt.xlabel('Sample loss')
	plt.tight_layout()

	output_dir = kwargs.get('output_dir', './')
	rel_fname = kwargs.get('rel_fname', 'loss_hist.png')
	fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
	plt.savefig(fname)

	if kwargs.get('show_plot', True):
		plt.show()


def plot_N_lossiest(input_data, output_data, **kwargs):


	print('Data shape: ', input_data.shape)


	mse = np.mean((input_data - output_data)**2, axis=1)

	print('MSE shape:', mse.shape)
	N = 15

	#N_lossiest_indices = mse.argsort()[-N:][::-1]
	N_lossiest_indices = mse.argsort()[:N][::-1]


	print(N_lossiest_indices)
	print('N lossiest MSEs: ', mse[N_lossiest_indices])

	data_size = input_data.shape[1]
	img_side_size = int(np.sqrt(data_size))

	input_data = input_data.reshape(-1, img_side_size, img_side_size)
	output_data = output_data.reshape(-1, img_side_size, img_side_size)

	N_lossiest_input = input_data[N_lossiest_indices]
	N_lossiest_output = output_data[N_lossiest_indices]


	plot_reconstructions(N_lossiest_input, N_lossiest_output, **kwargs)




def smooth_data(in_dat):

	'''
	Useful for smoothing data, when you have a ton of points and want fewer,
	or when it's really noisy and you want to see the general trend.

	Expects in_dat to just be a long list of values. Returns a tuple of
	the downsampled x and y, where the x are the indices of the y values,
	so you can easily plot the smoothed version on top of the original.
	'''

	hist = np.array(in_dat)
	N_avg_pts = min(100, len(hist)) #How many points you'll have in the end.

	avg_period = max(1, len(hist) // max(1, N_avg_pts))

	downsampled_x = avg_period*np.array(range(N_avg_pts))
	hist_downsampled_mean = np.array([hist[i*avg_period:(i+1)*avg_period].mean() for i in range(N_avg_pts)])
	return downsampled_x, hist_downsampled_mean



#
