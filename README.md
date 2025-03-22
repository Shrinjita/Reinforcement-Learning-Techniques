# ReinforcementLearningBasics

This repository contains implementations of foundational reinforcement learning experiments, inspired by the exercises and examples from Sutton and Barto's book, "Reinforcement Learning: An Introduction" (2nd Edition). Each experiment is designed to elucidate core concepts in reinforcement learning through practical coding exercises.

## Experiments Overview

1. **TIC TAC TOE.ipynb**
   - **Description**: Implements a reinforcement learning agent to play the classic game of Tic-Tac-Toe. The agent learns optimal strategies through self-play, utilizing concepts such as state-value functions and policy improvement.
   - **Key Concepts**: State representation, policy evaluation, exploration vs. exploitation.

2. **2_Bandit_Problem_Student_Copy.ipynb**
   - **Description**: Explores the k-armed bandit problem, a fundamental example in reinforcement learning that illustrates the trade-off between exploration and exploitation. Various action-value methods, including ε-greedy and optimistic initial values, are implemented and compared.
   - **Key Concepts**: Action-value methods, exploration strategies, regret minimization.

3. **3_MDP_Grid_World.ipynb**
   - **Description**: Models a grid world environment as a Markov Decision Process (MDP). The notebook demonstrates how to compute state-value functions and derive optimal policies using dynamic programming techniques.
   - **Key Concepts**: Markov Decision Processes, dynamic programming, Bellman equations.

4. **4_cleaning_robot.ipynb**
   - **Description**: Simulates a cleaning robot operating in a stochastic environment. The robot learns to navigate and clean efficiently using reinforcement learning algorithms, adapting to uncertainties in the environment.
   - **Key Concepts**: Stochastic environments, policy learning, reward structures.

5. **5_Gridworld_using_policy_iteration.ipynb**
   - **Description**: Implements policy iteration to solve the grid world problem. The notebook showcases the iterative process of policy evaluation and improvement to converge to the optimal policy.
   - **Key Concepts**: Policy iteration, convergence, optimal policies.

6. **6_Policy_And_Value_Iteration_Gambler_Problem.ipynb**
   - **Description**: Addresses the Gambler's Problem, where a gambler aims to reach a target wealth by betting on coin flips. Both policy iteration and value iteration methods are applied to determine the optimal betting strategy.
   - **Key Concepts**: Value iteration, policy iteration, convergence analysis.

7. **7_Gambler_Using_Value_Iteration.ipynb**
   - **Description**: Focuses solely on applying value iteration to solve the Gambler's Problem, providing insights into the convergence behavior and computational aspects of the algorithm.
   - **Key Concepts**: Value iteration, computational efficiency, policy extraction.

8. **Lab_10_Perform_compression_on_mnist_dataset_using_auto_encoder.ipynb**
   - **Description**: Utilizes autoencoders, a type of neural network, to perform data compression on the MNIST dataset. The experiment demonstrates how unsupervised learning can be applied to reduce dimensionality while preserving essential features.
   - **Key Concepts**: Autoencoders, dimensionality reduction, unsupervised learning.

9. **Lab_11_Build_a_Recurrent_Neural_Network.ipynb**
   - **Description**: Constructs a Recurrent Neural Network (RNN) to handle sequential data. The notebook illustrates the application of RNNs in tasks such as time-series prediction and natural language processing.
   - **Key Concepts**: Recurrent Neural Networks, sequence modeling, temporal dependencies.

10. **Lab_12_Implement_a_Deep_Convolutional_GAN_to_generate_images.ipynb**
    - **Description**: Implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic images. The experiment showcases the capability of GANs to learn complex data distributions and produce realistic outputs.
    - **Key Concepts**: Generative Adversarial Networks, convolutional networks, image generation.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd Edition). MIT Press. [Available online](https://incompleteideas.net/book/RLbook2020.pdf)

- Official code repository for the book: [Code for Sutton & Barto Book: Reinforcement Learning: An Introduction](https://incompleteideas.net/book/code/code2nd.html)

- Python replication of the book's examples: [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)

## Acknowledgments

This repository is inspired by the foundational work of Richard S. Sutton and Andrew G. Barto in the field of reinforcement learning. Their contributions have significantly advanced our understanding and application of reinforcement learning algorithms.
