# X-VLA
Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model

[Our paper](https://arxiv.org/pdf/2510.10274) | [Project page](https://thu-air-dream.github.io/X-VLA/)


https://github.com/user-attachments/assets/c047bac4-17c3-4d66-8036-badfab2b8c41



## Results on Simulations
We evluate X-VLA across 6 simulations, which encompass hundreds of evaluation setups, spanning single-arm, bi-manual robotic systems, autonomous driving, and assessing diverse axes of generalization, including cross-embodiment, cross-environment, and cross-task adaptation.

|Simpler|||Libero|||||Calvin|RoboTwin_2.0||VLABench|NAVSIM|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| VM | VM | WidowX |Spatial|Object|Goal|Long|Avg|ABC&rightarrow;D|Easy|Hard|Avg. PS|PDMS|
| 80.4 | 75.7 | 95.8 | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | 4.43 | 70.0 | 39.0 | 51.1 | 87.3 |

More detailed metrics for each benchmark are available in the following figures. Click to view: [Robotics Simulation](https://github.com/2toinf/X-VLA/blob/main/images/robo.jpeg) and [Autonomous Driving](https://github.com/2toinf/X-VLA/blob/main/images/ad.jpeg). 

### Server-Client Setup
Following [π₀](https://github.com/Physical-Intelligence/openpi), we adopt a server-client setup for simulation evaluations. Specifically, we run the policy and the simulation environment in separate Python processes, using a network-based server-client setup to enable communication between them. The policy acts as the server, while the simulation environment queries it as a client.

To start the server, run the following commands:

```
conda activate xvla
bash scripts/depoly.sh
```

Next, run the evaluation scripts to test the policy. To evaluate across different simulations, make sure to set up the corresponding environments: [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO), [Simpler](https://github.com/simpler-env/SimplerEnv), [Calvin](https://github.com/mees/calvin), [VLABench](https://github.com/OpenMOSS/VLABench), and [NAVSIM](https://github.com/autonomousvision/navsim).

For example, to evaluate the policy on the Libero benchmark, you can run the following commands after [installing Libero in a separate Conda environment](https://github.com/Lifelong-Robot-Learning/LIBERO):
```
conda activate libero
bash eval/libero/client.sh
```

#### More details on Libero
To enable the use of the Abs EEF as the control interface on LIBERO, we replay the dataset to obtain the corresponding actions:
```python
for action in actions:
    obs, reward, done, info = env.step(action)
    abs_pos = env.env.robots[0].controller.goal_pos
    abs_ori = env.env.robots[0].controller.goal_ori
```
For evaluation, the controller needs to be set to absolute control mode:
```python
env.reset()
for robot in env.env.robots:
robot.controller.use_delta=False
```
