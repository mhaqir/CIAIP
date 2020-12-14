# Written by Mohammad Haghir Ebrahimabdi
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from ple.games.pong import Pong
from ple import PLE
import pygame
import random
import numpy as np
import sys

from utils import agent, trail, extract_pad

NB_FRAMES = int(sys.argv[1])
NB_TRAILS = int(sys.argv[2])
DATA_DIR = 'data'

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

game = Pong(width=64, height=64)
p = PLE(game, fps=30, display_screen=False, force_fps=False)
action_set = p.getActionSet()

for nt in range(NB_TRAILS):
    random_generated_int = random.randint(0, 2**31-1)
    filename = DATA_DIR+"/"+str(random_generated_int)+".npz"
    p.init()
    obs_b_p = trail(NB_FRAMES, p, action_set)
    game_frames = np.array(obs_b_p, dtype=np.uint8)/255.0
    # np.savez_compressed(filename, obs_w_p=recording_obs_b_p)
    ball_frames = np.zeros_like(game_frames[1:, 7:58, :]) # drop the first frame
    ball_frames[:,:,:] = game_frames[1:,7:58,:]
    # print(np.shape(ball_frames))
    # print(np.shape(game_frames))
    # Extracting upper pads
    upper_pads = np.zeros_like(ball_frames)
    upper_pads[:, 2:7,:] = game_frames[1:, 2:7,:]
    # print(np.shape(upper_pads))
    upper_pads_o_b = extract_pad(upper_pads)
    
    # Extracting lower pads
    lower_pads = np.zeros_like(ball_frames)
    lower_pads[:, 45:50, :] = game_frames[1:, 58:63, :]
    # print(np.shape(lower_pads))
    # lower_pads[:, 58:63, :] = game_frames[1:, 58:63, :]
    lower_pads_o_b = extract_pad(lower_pads)

    np.savez_compressed(filename, ball = ball_frames, u_pad=upper_pads_o_b,\
	                    l_pad = lower_pads_o_b, whole_frames = game_frames[1:, :, :])