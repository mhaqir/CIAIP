

import matplotlib.animation as animation
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
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
frame = game_frames[95].reshape(1, 64, 64, 3)


z_size = 32
model_path_name = '/N/u/mhaghir/Carbonate/CogIA_project/CIAIP/VAE/vae'
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, 'vae_self_10.json'))

batch_z = vae.encode(frame)
# mu, logvar = vae.encode_mu_logvar(frame)
reconstruct = vae.decode(batch_z)


# Ploting a frame and its reconstructed output
# plt.imsave('output/orig_self.png', frame[0])
# plt.imsave('output/rec_self.png', reconstruct[0])


# Just random thoughts on visualizing the latent space
# argmin_mu = np.argmin(mu)
# argmax_mu = np.argmax(mu)

# grid = np.arange(mu[0, argmin_mu] - 3*np.exp(logvar[0, argmin_mu]/2), mu[0, argmax_mu] + 3*np.exp(logvar[0, argmax_mu]/2), 0.01)
# d = np.zeros_like(grid)

# for item in zip(mu[0], np.exp(logvar[0]/2)):
# 	print(item)
# 	d += norm.pdf(grid, loc = item[0], scale = item[1])
# print(d)

# f, ax = plt.subplots(1, figsize=(10, 7), sharex=False)
# ax.plot(grid, d)
# f.savefig('Latent_space_distribution.pdf', bbox_inches='tight')


# Animating a trail
# Fig = plt.figure('trail')
# img = [[plt.imshow(frame)] for frame in game_frames]
# anim = animation.ArtistAnimation(fig, img, interval = 20, blit = True, repeat_delay=0)
# anim.save('random_trail.mp4')



