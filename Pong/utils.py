# Written by Mohammad Haghir Ebrahimabdi
from copy import deepcopy
import random
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
from itertools import product

def agent(action_set):
  return random.choice(action_set)

def trail(nb_frames, p, action_set):
  obs_b_p = []
  for f in range(nb_frames):
    if p.game_over():
      p.reset_game()
    frame_b_p = p.getScreenGrayscale()
    obs_b_p.append(frame_b_p)
    action = agent(action_set)
    r = p.act(action)
  return obs_b_p

def extract_pad(f):
  frames = deepcopy(f)
  for i in range(np.shape(frames)[0]):
    pad_sum = np.sum(frames[i], axis = 0)
    pad_sum_5 = np.where(np.sum(frames[i], axis = 0) == 5)[0] # 5 is paddle height
    idx = []
    for j in range(np.shape(pad_sum_5)[0] - 1):
      if pad_sum_5[j + 1] == pad_sum_5[j] + 1:
        idx.append(pad_sum_5[j])
      else:
        if len(idx) > 8: # A threshold, should be a portion of the paddle length (here paddle length is 10)
          break
        else:
          idx = []
    while len(idx) < 10: # paddle length
      if idx[-1] < 63:  # frame width
        if pad_sum[idx[-1] + 1] > 0:
          idx.append(idx[-1] + 1)
        else:
          idx.insert(0,idx[0] - 1)
    frames[i, :, [k for k in range(np.shape(frames)[2]) if k not in idx]] = 0
  return frames

def animate(frames, anim_name):
  fig = plt.figure('fig')
  imgs = [[plt.imshow(f, cmap='gray')] for f in frames]
  anim = animation.ArtistAnimation(fig, imgs, interval = 30, blit = True, repeat_delay=0)
  anim.save('{}.mp4'.format(anim_name))


def rec_multiple_frames_plot(nrows, nclos, x_limit, y_limit, x_step, y_step, vae): # x_limit and y_limit are both lists containing two elements, and the nrows and ncols must
																			# be calculated based on the limits and step sizes
	rc = {"axes.spines.left" : False,
	      "axes.spines.right" : False,
	      "axes.spines.bottom" : False,
	      "axes.spines.top" : False,
	      "xtick.bottom" : False,
	      "xtick.labelbottom" : False,
	      "ytick.labelleft" : False,
	      "ytick.left" : False}
	plt.rcParams.update(rc)

	step1 = np.arange(y_limit[0], y_limit[1], x_step)
	step2 = np.arange(x_limit[0], x_limit[1], y_step)
	xy = list(product(step1, step2))

	fig, axs = plt.subplots(nrows = nrows, ncols = nclos, figsize = (15, 15))
	axs = axs.ravel()
	for i in range(len(xy)):
	  batch_z = np.expand_dims(np.asarray(xy[i]), 0)
	  # reconstruct = vae.decode(batch_z)
	  axs[i].imshow(np.squeeze(vae.decode(batch_z)), cmap = 'gray')
	fig.savefig('z_explore.pdf', bbox_inches='tight')
	plt.close()


def create_dataset(filelist, N, data_dir): # N is the number of trails
  data = []
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(data_dir, filename))['ball']
    raw_data_o_0 = [d for d in raw_data if np.sum(d) != 0]
    data += raw_data_o_0
  return np.expand_dims(np.asarray(data, dtype=np.uint8), -1)


