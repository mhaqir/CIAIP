#!/bin/python3


import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy
import tensorflow as tf
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae import ConvVAE, reset_graph

DATA_DIR = '/N/u/mhaghir/Carbonate/CogIA_project/CIAIP/VAE/Data'

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[195:]


data = np.load(DATA_DIR + '/' + filelist[1])
game_frames = data['obs']
game_frames = game_frames.astype(np.float32)/255.0

# frame = random.choice(frames).reshape(1, 64, 64, 3)
frame = game_frames[5].reshape(1, 64, 64, 3)

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
# ax.imshow(frame[0])
# fig.savefig('frame.pdf')
# plt.close()


z_size = 2
model_path_name = '/N/u/mhaghir/Carbonate/CogIA_project/CIAIP/VAE/vae'
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

# vae.load_json(os.path.join(model_path_name, 'vae_self_10_zsize_2.json'))

batch_z = vae.encode(frame[:, 0:52, :])

# mu, logvar = vae.encode_mu_logvar(frame)

# fig, axs = plt.subplots(nrows = 8, ncols = 4, figsize = (14, 15))
# axs = axs.ravel()
# for i in range(32):
# 	# z_copy = deepcopy(batch_z)
# 	# z_copy[0, i] == 0
# 	z_copy = np.expand_dims(np.random.random_sample((2,)), 0)
# 	axs[i].imshow(np.squeeze(vae.decode(z_copy)))
# 	# axs[i].legend( 'dim_{}'.format(i), loc='lower right')
# 	# axs[i].set_ylim(0.8,1)
# fig.savefig('z_explore.pdf')
# plt.close()



# reconstruct = vae.decode(batch_z)
# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
# ax.imshow(reconstruct[0])
# fig.savefig('rec_frame.pdf')
# plt.close()


# model_params, model_shapes, model_names = vae.get_model_params()
# for param in zip(model_snames, model_shapes):
# 	print(param[0], ': ', param[1])
# 	print('\n')

h, h1, h2, h3, h4, h5, h6, h7, h8, h9, y = vae.hidden(frame[:, 0:52, :])
print("h: ", np.shape(h))
print("h1: ", np.shape(h1))
print("h2: ", np.shape(h2))
print("h3: ", np.shape(h3))
print("h4: ", np.shape(h4))
print("h5: ", np.shape(h5))
print("h6: ", np.shape(h6))
print("h7: ", np.shape(h7))
print("h8: ", np.shape(h8))
print("h9: ", np.shape(h9))
print("y: ", np.shape(y))



