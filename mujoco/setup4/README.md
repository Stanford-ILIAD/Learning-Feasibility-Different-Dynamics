# The code is based on 'https://github.com/ikostrikov/pytorch-trpo'. Instructions to run the code including train the optimal policy for f-MDP and imitation learning from reweighted demonstrations.

## Train Feasibility

### Swimmer

```
python main_feasibility.py --save_path checkpoints/swimmer_feasibility_setup --demo_files ../demo/swimmer_back_disable_12/batch_00.pkl ../demo/swimmer_back_disable_20/batch_00.pkl ../demo/swimmer/batch_00.pkl ../demo/swimmer_front_disable_10/batch_00.pkl --test_demo_files ../demo/swimmer_back_disable_12/batch_00.pkl ../demo/swimmer_back_disable_20/batch_00.pkl ../demo/swimmer/batch_00.pkl ../demo/swimmer_front_disable_10/batch_00.pkl --xml back_leg_disable.xml  --env-name CustomSwimmerFeasibility-v0 --ratio 0.1 0.1 0.1 0.1 --mode traj --discount_train --discount 0.99
```

### Walker2d Setup2

```
python main_feasibility.py --save_path checkpoints/walker2d_feasibility_setup2 --demo_files ../demo/walker2d_0.7/batch_00.pkl ../demo/walker2d_1.1/batch_00.pkl ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_29.9/batch_00.pkl --test_demo_files ../demo/walker2d_0.7/batch_00.pkl ../demo/walker2d_1.1/batch_00.pkl ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_29.9/batch_00.pkl --xml walker2d.xml  --env-name CustomWalker2dFeasibility-v0 --ratio 0.02 0.02 0.02 0.02 --mode traj --discount_train --batch-size 25000
```

### HalfCheetah Setup1

```
python main_feasibility.py --save_path checkpoints/half_cheetah_feasibility_setup1 --demo_files ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_front_0.5/batch_00.pkl ../demo/half_cheetah_back_0.5/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl --test_demo_files ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_front_0.5/batch_00.pkl ../demo/half_cheetah_back_0.5/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl --xml half_cheetah_front_0.01.xml  --env-name CustomHalfCheetahFeasibility-v0 --ratio 0.05 0.05 0.05 0.05 --mode traj --discount_train
```

### HalfCheetah Setup2

```
python main_feasibility.py --save_path checkpoints/half_cheetah_feasibility_setup1 --demo_files ../demo/half_cheetah_front_0.01/batch_00.pkl ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl ../demo/half_cheetah_back_0.01/batch_00.pkl --test_demo_files ../demo/half_cheetah_front_0.01/batch_00.pkl ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl ../demo/half_cheetah_back_0.01/batch_00.pkl --xml half_cheetah_front_0.01.xml  --env-name CustomHalfCheetahFeasibility-v0 --ratio 0.05 0.05 0.05 0.05 --mode traj --discount_train
```

## Reweighted Imitation

### Swimmer

```
python main_gailfo.py --env-name CustomSwimmer-v0 --demo_files ../demo/swimmer_back_disable_12/batch_00.pkl ../demo/swimmer_back_disable_20/batch_00.pkl ../demo/swimmer/batch_00.pkl ../demo/swimmer_front_disable_10/batch_00.pkl --save_path checkpoints/swimmer_imitate.pth --eval-interval 5 --num-epochs 20000 --ratios 0.1 0.1 1 1 --xml back_leg_disable.xml --feasibility_model checkpoints/swimmer_feasibility_setup --mode traj --discount 0.99
```

### Walker2d Setup2

```
python main_gailfo.py --env-name CustomWalker2d-v0 --demo_files ../demo/walker2d_0.7/batch_00.pkl ../demo/walker2d_1.1/batch_00.pkl ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_29.9/batch_00.pkl --save_path checkpoints/walker2d_imitate2.pth --eval-interval 5 --num-epochs 20000 --ratios 0.02 0.02 1 1 --xml walker2d.xml --feasibility_model checkpoints/walker2d_feasibility_setup2 --mode traj
```

### HalfCheetah Setup1

```
python main_gailfo.py --env-name CustomHalfCheetah-v0 --demo_files ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_front_0.5/batch_00.pkl ../demo/half_cheetah_back_0.5/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl --save_path checkpoints/half_cheetah_imitate1.pth --eval-interval 5 --num-epochs 20000 --ratios 0.05 1 1 1 --xml half_cheetah_front_0.01.xml --feasibility_model checkpoints/half_cheetah_feasibility_setup1 --mode traj
```

### HalfCheetah Setup2

```
python main_gailfo.py --env-name CustomHalfCheetah-v0 --demo_files ../demo/half_cheetah_front_0.01/batch_00.pkl ../demo/half_cheetah_front_0.05/batch_00.pkl ../demo/half_cheetah_back_0.05/batch_00.pkl ../demo/half_cheetah_back_0.01/batch_00.pkl --save_path checkpoints/half_cheetah_imitate2.pth --eval-interval 5 --num-epochs 20000 --ratios 0.05 0.05 1 1 --xml half_cheetah_front_0.01.xml --feasibility_model checkpoints/half_cheetah_feasibility_setup2 --mode traj
```
