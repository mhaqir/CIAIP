#!/bin/python3
# Written by Mohammad Haghir Ebrahimabadi

VAE_DIR = '/N/u/mhaghir/Carbonate/CogIA_project/WorldModelsExperiments/carracing/vae'

import subprocess
# subprocess.call(['ln', '-s',  VAE_DIR + '/vae.py', '.']) # create soft links
# subprocess.call(['ln', '-s',  VAE_DIR + '/vae.json', '.'])
import tensorflow as tf
from vae import ConvVAE

z_size = 32

vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json('vae.json')

# graph = tf.get_default_graph()
# operations = graph.get_operations()
# print(operations)
# layers = vae.layer()
# vae.summary()
# print(vae)
