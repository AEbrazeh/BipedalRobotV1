[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# 4-DoF Bipedal Robot V1

# Design, Simulation and Control of a 4-DoF Bipedal Robot

This repository contains all the materials related to my BSc final project at the University of Tehran. The project revolves around the design, simulation, and control of a **4-DoF bipedal robot** that emulates the human walking cycle.

## Project Overview

The bipedal robot was designed using **Solidworks** and simulated to mimic the human walking cycle. The robot's control system was governed by an **[Intelligent PID (iPID) controller](https://inria.hal.science/inria-00273279v1/)** and a **[TD3 Algorithm](https://arxiv.org/abs/1802.09477v3)**, which together facilitated the generation of a natural and stable walking motion.

The model was capable of producing diverse walk cycles, characterized by varying cycle periods and stance phase periods. To ensure balance, the robot was equipped with **three reaction wheels**, all of which were regulated by the iPID controller. A key feature of the model was its resilience to external disturbances.

<img src="https://github.com/AEbrazeh/BipedalRobotV1/blob/main/gifs/View1.gif" width="500" height="500"/>
<img src="https://github.com/AEbrazeh/BipedalRobotV1/blob/main/gifs/View2.gif" width="500" height="500"/>

## Key Features

* **Intelligent Proportional-Integral-Derivative (iPID) Control**: Fine-tuned PID parameters for stable balancing and regulation of the reaction wheels.
* **Reinforcement Learning**: Utilized Twin Delayed DDPG (TD3) algorithm for efficient and robust walking patterns.
* **Simulation**: Implemented in PyBullet for realistic physics and accurate modeling.
* **Python Packages**: Leveraged PyTorch for neural network implementation and NumPy for numerical computations.

This project showcases the application of advanced control systems in the development of a bipedal robot capable of maintaining balance and walking, demonstrating resilience to external disturbances.vvv

## Versioning
This repository represents version 1 (v1) of the project. Future enhancements and iterations will be released as new versions.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
