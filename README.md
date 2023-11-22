# sequence-learning-RL
RL model for sequence learning, with several type of algorithms

## Deep Q-Network Algorithm
The `optimize_model` function is a crucial component of the Deep Q-Network (DQN) algorithm used in reinforcement learning. This function is responsible for updating the policy network based on experiences (transitions) stored in the replay memory.

## Important Variables and Concepts

- `BATCH_SIZE`: Determines the number of experiences sampled from the replay memory for each update.
- `GAMMA` (Discount Factor): Used to balance the importance of immediate and future rewards. Typically set between 0 and 1.
- `state_batch`: A batch of observed states.
- `action_batch`: A batch of actions taken in those states.
- `reward_batch`: A batch of rewards received for taking those actions.
- `state_action_values`: The Q-values predicted by the policy network for the state-action pairs.
- `next_state_values`: The Q-values for the next states, predicted by the target network.
- `expected_state_action_values`: The target Q-values that the policy network is trained to predict.

## Mathematics Involved

The key mathematical concept in `optimize_model` is the computation of the loss between the Q-values predicted by the policy network and the target Q-values. The target Q-values are calculated using the Bellman equation:

### Q-values
$$\text{expected\_state\_action\_values} = (\text{next\_state\_values} \times \gamma) + \text{reward\_batch}$$

### Loss Calculation

The Huber loss (smooth L1 loss) is used for training:

$\text{loss} = \text{smooth\_l1\_loss}(\text{state\_action\_values}, \text{expected\_state\_action\_values})$

$L_\delta(a) =\begin{cases}\frac{1}{2} a^2 & \text{for } |a| \le \delta \\\delta (|a| - \frac{1}{2} \delta) & \text{otherwise}\end{cases}$


This process involves zeroing the gradients, performing backpropagation, and then updating the weights with an optimizer (e.g., Adam).

## References
- [PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Deep Q-Networks (DQN) Introduction](https://example-link-to-dqn-intro.com)
- [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto](https://example-link-to-sutton-barto-book.com)
- [Huber Loss Explanation](https://example-link-to-huber-loss.com)

For a detailed understanding of DQN and its implementation, refer to the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al., available at [Link to Mnih et al. paper](https://example-link-to-mnih-paper.com).