
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import plot_utils
import os
import shutil

#run_label = 'mnist'
#run_label = 'dogpeople_vae_0KL'
#run_label = 'dogpeople_vae_0KL_1e-8Adam'
#run_label = 'mnist_vae_0pt0001KL_ramp'
#run_label = 'mnist_vae_0pt0001KL'
#run_label = 'mnist_vae_0KL'
#run_label = 'current_running_VAE'

#run_label = 'dogpeople_vae_0KL1500hidden_50latent'
run_label = 'dogpeople_vae_0KL_train_with_friends2000hidden_60latent'

base_source_dir = 'output'
source_dir = os.path.join(base_source_dir, run_label)

base_output_dir = 'output_pics'
output_dir = os.path.join(base_output_dir, run_label)

if os.path.exists(output_dir):
    print(f'Removing dir {output_dir}...')
    shutil.rmtree(output_dir)

print(f'Creating dir {output_dir}...')
os.mkdir(output_dir)


print('Reading in data...')
in_data_raw = genfromtxt(os.path.join(source_dir, run_label + '_input_batch.txt'), delimiter=',')
out_data_raw = genfromtxt(os.path.join(source_dir, run_label + '_output_batch.txt'), delimiter=',')
latent_data = genfromtxt(os.path.join(source_dir, run_label + '_latent_batch.txt'), delimiter=',')

labels = genfromtxt(os.path.join(source_dir, run_label + '_labels.txt'), delimiter=',')

latent_path = genfromtxt(os.path.join(source_dir, run_label + '_latent_path.txt'), delimiter=',')
latent_path_recon = genfromtxt(os.path.join(source_dir, run_label + '_latent_path_recon.txt'), delimiter=',')


latent_grid = genfromtxt(os.path.join(source_dir, run_label + '_latent_grid.txt'), delimiter=',')
latent_grid_recon = genfromtxt(os.path.join(source_dir, run_label + '_latent_grid_recon.txt'), delimiter=',')

'''
print('in_data: {}'.format(in_data.shape))
print('out_data: {}'.format(out_data.shape))
print('latent_data: {}'.format(latent_data.shape))
print('labels: {}'.format(labels.shape))
print('latent_grid: {}'.format(latent_grid.shape))
print('latent_path: {}'.format(latent_path.shape))
'''


data_size = in_data_raw.shape[1]
latent_size = latent_data.shape[1]
img_side_size = int(np.sqrt(data_size))

in_data = in_data_raw.reshape(-1, img_side_size, img_side_size)
out_data = out_data_raw.reshape(-1, img_side_size, img_side_size)

latent_path_recon = latent_path_recon.reshape(-1, img_side_size, img_side_size)

side = int(np.sqrt(latent_grid_recon.shape[0]))
latent_grid_recon = latent_grid_recon.reshape(side, side,  img_side_size, img_side_size)



show_plot = False
print('Plotting...')


if 'current_running' not in run_label:

    losses_kl = genfromtxt(os.path.join(source_dir, run_label + '_losses_kl.txt'), delimiter=',')
    beta_kl = genfromtxt(os.path.join(source_dir, run_label + '_beta_KL.txt'), delimiter=',')
    losses_recon = genfromtxt(os.path.join(source_dir, run_label + '_losses_recon.txt'), delimiter=',')
    losses_total = genfromtxt(os.path.join(source_dir, run_label + '_losses_total.txt'), delimiter=',')
    loss_dict = {
        'beta_KL' : {'data' : beta_kl, 'ylabel' : r'$\beta_{KL}$', 'yscale' : 'linear', 'xlabel' : 'Epoch'},
        'losses_kl' : {'data' : losses_kl, 'ylabel' : 'KL divergence loss', 'yscale' : 'linear'},
        'losses_recon' : {'data' : losses_recon, 'ylabel' : 'Reconstruction loss'},
        'losses_total' : {'data' : losses_total, 'ylabel' : 'Total loss'},
    }


    plot_utils.plot_losses(loss_dict, output_dir=output_dir, show_plot=show_plot)






plot_utils.plot_reconstructions(in_data, out_data, output_dir=output_dir, show_plot=show_plot)

plot_utils.plot_transformation(latent_path_recon, output_dir=output_dir, show_plot=show_plot)


plot_utils.plot_2D_latent_space(latent_data[:,:2], label_list=labels, output_dir=output_dir, show_plot=show_plot, rel_fname='latent_space')
plot_utils.plot_2D_latent_space(latent_data[:,:2], label_list=labels, highlight_points=latent_path[:,:2], output_dir=output_dir, show_plot=show_plot, rel_fname='latent_path')

plot_utils.plot_2D_latent_space(latent_data[:,:2], label_list=labels, highlight_points=latent_grid[:,:2], output_dir=output_dir, show_plot=show_plot, rel_fname='latent_grid')

plot_utils.plot_image_grid(latent_grid_recon, output_dir=output_dir, show_plot=show_plot)





in_data_friend_raw = genfromtxt(os.path.join(source_dir, run_label + '_friend_input_batch.txt'), delimiter=',')
out_data_friend_raw = genfromtxt(os.path.join(source_dir, run_label + '_friend_output_batch.txt'), delimiter=',')
in_data_friend = in_data_friend_raw.reshape(-1, img_side_size, img_side_size)
out_data_friend = out_data_friend_raw.reshape(-1, img_side_size, img_side_size)

plot_utils.plot_reconstructions(in_data_friend, out_data_friend, output_dir=output_dir, show_plot=show_plot, rel_fname='friend_recons')


plot_utils.plot_N_lossiest(in_data_raw, out_data_raw, output_dir=output_dir, rel_fname='N_worst.png', show_plot=show_plot)
plot_utils.plot_losses_hist(in_data_raw, out_data_raw, output_dir=output_dir, rel_fname='sample_losses_hist.png', highlight_pts={'in_pts':in_data_friend_raw, 'out_pts':out_data_friend_raw}, show_plot=show_plot)






friend_names = ['ben', 'bobby', 'david', 'liz', 'max', 'phil', 'will']
friend_interps = [genfromtxt(os.path.join(source_dir, run_label + '_' + name + '_interp.txt'), delimiter=',').reshape(-1, img_side_size, img_side_size) for name in friend_names]

[plot_utils.create_interp_gif(friend_interps[i], output_dir, 'transform_gif_{}.gif'.format(friend_names[i])) for i in range(len(friend_names))]


[plot_utils.plot_transformation(interp, output_dir=output_dir, show_plot=False, rel_fname='transform_{}'.format(friend_names[i])) for i,interp in enumerate(friend_interps)]


exit()



#
