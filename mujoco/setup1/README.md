# The code is built based on 'https://github.com/ikostrikov/jaxrl'. Instructions to run the code including train the optimal policy for f-MDP and imitation learning from reweighted demonstrations.

## Train Feasibility

### Walker2d Setup1

```
python main_feasibility.py --save_path checkpoints/walker2d_feasibility_setup1 --demo_files ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl --test_demo_files ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl --xml walker2d_24.9.xml  --env-name CustomWalker2dFeasibility-v0 --ratio 0.1 0.1 0.1 0.1 --mode traj --discount_train
```

## Reweighted Imitation

### Walker2d Setup1

```
python main_gailfo.py --env-name CustomWalker2d-v0 --demo_files ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl --save_path checkpoints/walker2d_imitate1.pth --eval-interval 5 --num-epochs 20000 --ratios 0.1 0.1 1 1 --xml walker2d_24.9.xml --feasibility_model checkpoints/walker2d_feasibility_setup1 --mode traj
```

