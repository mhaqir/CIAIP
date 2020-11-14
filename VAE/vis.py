

import numpy as np
import random
import os
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae import ConvVAE, reset_graph

DATA_DIR = '/N/u/mhaghir/Carbonate/CogIA_project/CIAIP/VAE/Data'

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[195:]

print(filelist)

data = np.load(DATA_DIR + '/' + filelist[1])
frames = data['obs']
frames = frames.astype(np.float32)/255.0
# frame = random.choice(frames).reshape(1, 64, 64, 3)
frame = frames[95].reshape(1, 64, 64, 3)


z_size = 32
model_path_name = '/N/u/mhaghir/Carbonate/CogIA_project/CIAIP/VAE/vae'
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, 'vae_self_10.json'))

batch_z = vae.encode(np.expand_dims(frame[0], 0))
print(batch_z[0]) # print out sampled z
# print(np.shape(batch_z[0]))
reconstruct = vae.decode(batch_z)


plt.imsave('orig_self.png', frame[0])
plt.imsave('rec_self.png', reconstruct[0])
