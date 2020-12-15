# Data

The list of packages installed on the system that used is provided in the packages.txt file. For installing the specific requirements for this project such as a working version of the pygame package the dep.sh file should run.

```console

foo@bar:~$ ./dep.sh

```
Note that if you want to increase the size of the paddles and the ball, you should install the PLE package separately after increasing the numbers in the lines 208 and 214 to 0.08 and 0.083 respectively in the PyGame-Learning-Environment/ple/games/pong.y file.

For generating a dataset, the data.py file can be used. Note that the number of game trails and the number of frames in each trail must be given in the command line when running this file:

```console

foo@bar:~$ python data.py <n_frames> <n_trails>

n_frames: number of frames in each game trail
n_trails: number of trials
```
please refer to [run_data.sh](https://github.com/mhaqir/CIAIP/blob/main/Pong/run_data.sh) for an example. You can also run the run_data.sh file for generating the data using the following line on IU Carbonate machine.
```console

foo@bar:~$ qsub -l nodes=1:ppn=4,vmem=16gb,walltime=02:00:00 run_data.sh
```
The data will be stored in 'data' folder in .npz format. Each .npz file contains the frames of one trail. In addition, the paddles and the ball will be extracted from each frame and be stored in the same .npz file.

# Beta-VAE

For training the VAE, the main.py file can be used.

```console
foo@bar:~$ python main.py <mode> <beta> <n_epochs>

mode: 'train' or 'test'
beta: beta value in beta-VAE (default is 1)
n_epochs: number of epochs
```
The 'test' mode can be used to load the model with the provided beta and n_epochs (these values are used in the name of the stored model) and generate a grid of reconstructed images by changing the latent space values.

# Attention
The .ipynb file in this directory contains the codes for running the experiments for square frames (64x64) which is explained as 'The first approach' in the report under Pong section. The reset of the codes are for the trimmed frames (51x64) corresponding to 'The second approach' under Pong section in the report.
