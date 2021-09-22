### Set up the environment

Needed packages: Python, torch, gym, pybullet, tensorflow, stable_baselines, random, numpy, math, pickle

### Data description(files in `data` folder)

Due to the space limit, we included the demonstrations [here](https://drive.google.com/drive/folders/1j7pHoRbdBvxZDfCSVROpVFbtVldbERie?usp=sharing). Download `data.zip`. And unzip it here.

`dis5.pkl`: contains 5 expert trajs for 3DoF

`normal48.pkl`: contains 48 expert trajs for 7DoF

`dis5ee.pkl`: contains 5 expert trajs for 3DoF (ee_position only)

`normal48ee.pkl`: contains 48 expert trajs for 7DoF(ee_position only)

`all_53.pkl`: contains all 53 expert trajs(5+48)

`dis5_3model`: rl feasibility model for dis5.pkl

`normal48_3model`: rl feasibility model for normal48.pkl

`weights_3.pkl`: the generated weight(feasibility) for 53 trajs(5+48)

`3_fea.pt`: feasibility VAE model

`3_normal.pt`: no feasibility VAE model

### How to create feasibility model
cd into `pdenv` directory, the file to create feasibility model is main_feasibility.py.

Important flags:

--env-name: Use 'feasibilitypanda-v0' as the environment name

--mode: Use 'dis' to create rl model for 3DoF robot arm, and use 'normal' to create rl model for 7DoF robot arm


Sample call:

`python main_feasibility.py --env-name "feasibilitypanda-v0" --seed 3 --save_path "../data" --discount 1 --mode 'dis' --disdata '../data/dis5.pkl' --normaldata '../data/normal48.pkl'`

The file to generate weights(feasibility) from the rl model is generate_weights.py, this will be used to create VAE model

Sample call:

`python generate_weights.py`

### How to create VAE model
cd into `SAIL` directory

Sample call:

To train VAE model with feasibility:

`python train_VAE.py --state-dim 3 --expert-traj-path ../data/all_53.pkl --weight True --weight-path ../data/weights_3.pkl --size-per-traj 4000 --output-path ../data/ --epoch 1 --seed 3`

To train VAE model without feasibility(baseline):

`python train_VAE.py --state-dim 3 --expert-traj-path ../data/all_53.pkl --weight False --size-per-traj 4000 --output-path ../data/ --epoch 1 --seed 3`

### Tips:

#### Disable or Enable GUI
`gym_panda/gym_panda/envs/panda_env.py` Line 19 and 20
```python
p.connect(p.DIRECT) # disable GUI
p.connect(p.GUI) # Enable GUI
```

#### Window/Linux
This version is able to run directly in Windows, change path in panda.py, generate_weights.py, and panda_env.py to use in Linux




