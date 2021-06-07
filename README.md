# IntelliHealer

[![Documentation Status](https://readthedocs.org/projects/intellihealer/badge/?version=latest)](https://intellihealer.readthedocs.io/en/latest/?badge=latest)

**IntelliHealer**: An imitation and reinforcement learning platform for 
self-healing distribution networks. IntelliHealer uses imitation learning framework to learn restoration policy 
for distribution system service restoration so as to perform the restoration 
actions (tie-line switching and reactive power dispatch) in real time and in 
embedded environment.

It is worth mentioning that the imitation lealrning framework acts as a bridge between reinforcement learning-based 
techniques and mathematical programming-based methods and a way to leverage well-studied mathematical programming-based 
decision-making systems for reinforcement learning-based automation.

|                                         Scope: Training restoration agent                                                   |                                               Framework: imitation learning                                          |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| <img src="project_dis_restoration/results/plots/Scope.png" width=100%>  | <img src="project_dis_restoration/results/plots/Algorithms.png" width=100%> |

Such **embeddable** and **computation-free** policies allows us to integrate the 
self-healing capability into intelligent devices 
A polit project conducted by the [S&C Electric](https://www.sandc.com/en/)
can be found [here](https://www.sandc.com/en/solutions/self-healing-grids/).
For details of this work, please refer to our paper at 
[arXiv](https://arxiv.org/abs/2011.14458/) 
or [IEEE](https://ieeexplore.ieee.org/document/9424985?denied=).

Features
=========
* **IntelliHealer proposes the imitation learning framework,** 
  which improve the sample efficiency using a mixed-integer program-based expert 
  compared with the traditional exploration-dominant reinforcement learning algorithms.
  
  <img src="project_dis_restoration/results/plots/fig_avg_ratio_comp_IL_RL_n_5.png" width=50%>
  
* **IntelliHealer proposes a hierarchical policy network,** 
  which can accommodate both discrete and continuous actions. 
  
  <img src="project_dis_restoration/results/plots/Hybrid_policy.png" width=60%>
 
* **IntelliHealer provides an [OpenAI-Gym](https://gym.openai.com/) environment for 
  distribution system restoration,** 
  which can be connected to [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/?badge=master), 
  a state-of-the-art collection of reinforcement learning algorithms. Currently, the Gym environment
  contains two test feeders: 33-node and 119-node system.
  
* **IntelliHealer provides distribution system optimization models built on [Pyomo](http://www.pyomo.org/documentation),**
  whicn can be used to develop other problem formulations.

Documentation
===============
For installation instructions, basic usage and benchmarks results, see the [official documentation](https://intellihealer.readthedocs.io/en/latest/).

Acknowledgments
=================
Based upon work supported by the **U.S. Department of Energy Advanced Grid Modeling Program** under Grant DE-OE0000875.


Citing
========

If you find this code useful in your research, please consider citing:
```
Y. Zhang, F. Qiu, T. Hong, Z. Wang, and F. Li, “Hybrid imitation learning for real-time service restoration in resilient distribution systems,” IEEE Trans. Ind. Informat., pp. 1-11,early access, 2021, doi: 10.1109/TII.2021.3078110.
```
```bibtex
@article{Zhang2021_IntelliHealer,
author = {Zhang, Yichen and Qiu, Feng and Hong, Tianqi and Wang, Zhaoyu and Li, Fangxing Fran},
journal = {IEEE Trans. Ind. Informat.},
keywords = {Deep learning,Imitation learning,Mixed-integer linear programming,Reinforcement learning,Resilient distribution system,Service restoration},
pages = {1--11},
note={early access},
title = {{Hybrid imitation learning for real-time service restoration in resilient distribution systems}},
year = {2021}
}
```

Related Works
=================
Regarding Imitation and Reinforcement Learning
------------------------------------------------
The framework development is based on the following work:
* Ross, Stéphane, and Drew Bagnell. "Efficient reductions for imitation learning." In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 661-668. JMLR Workshop and Conference Proceedings, 2010.
* Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627-635. JMLR Workshop and Conference Proceedings, 2011.
* Le, Hoang, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue, and Hal Daumé. "Hierarchical imitation and reinforcement learning." In International Conference on Machine Learning, pp. 2917-2926. PMLR, 2018.

The algorithm implementation is partially based on the work and its repository [hierarchical_IL_RL](https://github.com/hoangminhle/hierarchical_IL_RL): 
* Le, Hoang, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue, and Hal Daumé. "Hierarchical imitation and reinforcement learning." In International Conference on Machine Learning, pp. 2917-2926. PMLR, 2018.

Regarding Machine Learning for Optimization
---------------------------------------------
The proposed method can also be regarded as one of the three learn-to-optimize paradigms concluded in the following
literature:
* Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research (2020).

The three learn-to-optimize paradigms are illustrated below, where **our method serves as an end-to-end paradigm**:

<img src="project_dis_restoration/results/plots/Neural_MIP.PNG" width=90%>



License
-------
Released under the modified BSD license. See `LICENSE` for more details.
