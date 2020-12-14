# Data

For generating a dataset, the data.py file can be used. Note that the number of game trails and the number of frames in each trail must be given in the command line when running this file:

```console

foo@bar:~$ python data.py <n_frames> <n_trails>

n_frames: number of frames in each game trail
n_trails: number of trials
```
please refer to [run_data.sh](https://github.com/mhaqir/CIAIP/blob/main/Pong/run_data.sh) for an example.

The data will be stored in 'data' folder in .npz format. Each .npz file contains the frames of one trail. In addition, the paddles and the ball will be extracted from each frame and be stored in the same .npz file.

# Beta-VAE

For training the VAE, the main.py file can be used.

```console
foo@bar:~$ python main.py <mode> <beta> <n_epochs>

mode: 'train' or 'test'
beta: beta value in beta-VAE (default is 1)
n_epochs: number of epochs
```
