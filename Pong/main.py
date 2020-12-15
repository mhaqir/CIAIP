# Mainly written by Mohammad Haghir Ebrahimabdi
import os
import tensorflow as tf
from tqdm import tqdm
import sys
import numpy as np

from utils import create_dataset
from utils import rec_multiple_frames_plot
from vae import ConvVAE, reset_graph

z_size=2
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5
num_epoch = int(sys.argv[3])
beta = float(sys.argv[2])
DATA_DIR = 'data'

MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)

if sys.argv[1] == 'train':

	filelist = os.listdir(DATA_DIR)
	filelist.sort()
	filelist = filelist[0:199]
	dataset = create_dataset(filelist, 199, DATA_DIR)
	total_length = len(dataset)
	num_batches = int(np.floor(total_length/batch_size))

	reset_graph()

	vae = ConvVAE(z_size=z_size,
	              batch_size=batch_size,
	              learning_rate=learning_rate,
	              kl_tolerance=kl_tolerance,
	              is_training=True,
	              reuse=False,
	              gpu_mode=True,
	              beta = 1)

	# train loop:
	print("train", "step", "loss", "recon_loss", "kl_loss")
	for epoch in tqdm(range(num_epoch)):
	  np.random.shuffle(dataset)
	  for idx in range(num_batches):
	    batch = dataset[idx*batch_size:(idx+1)*batch_size]

	    obs = batch.astype(np.float)

	    feed = {vae.x: obs,}

	    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
	      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
	    ], feed)
	  
	    if ((train_step+1) % 500 == 0):
	      print("step", (train_step+1), train_loss, r_loss, kl_loss)
	    if ((train_step+1) % 5000 == 0):
	      vae.save_json(MODEL_DIR + "/vae_pong_ne_{ne}_beta_{beta}.json".format(ne=num_epoch, beta=beta))

	# finished, final model:
	vae.save_json(MODEL_DIR + "/vae_pong_ne_{ne}_beta_{beta}.json".format(ne=num_epoch, beta=beta))

if sys.argv[1] == 'test':

	vae = ConvVAE(z_size=z_size,
	              batch_size=1,
	              is_training=False,
	              reuse=False,
	              gpu_mode=False, 
	              beta=1)

	if os.path.exists(os.path.join(MODEL_DIR, 'vae_pong_ne_{ne}_beta_{beta}.json'.format(ne = num_epoch, beta = beta))):
		vae.load_json(os.path.join(MODEL_DIR, 'vae_pong_ne_{ne}_beta_{beta}.json'.format(ne = num_epoch, beta = beta)))
		rec_multiple_frames_plot(8, 8, [-2, 2], [-2, 2], 0.5, 0.5, vae)
	else:
		print("The model does not exist!")





