import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=64, help='the batch size of updating')  # Increased batch size for faster learning
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--buffer-size', type=int, default=50000, help='the size of the buffer')  # Reduced buffer size for faster training
    parse.add_argument('--cuda', action='store_true', help='if use the gpu')
    parse.add_argument('--init-ratio', type=float, default=1.0, help='the initial exploration ratio')
    parse.add_argument('--exploration_fraction', type=float, default=0.1, help='decide how many steps to do the exploration')
    parse.add_argument('--final-ratio', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--grad-norm-clipping', type=float, default=5.0, help='the gradient clipping')  # Reduced gradient clipping to avoid excessive updates
    parse.add_argument('--total-timesteps', type=int, default=int(1e6), help='the total timesteps to train network')  # Reduced total timesteps
    parse.add_argument('--learning-starts', type=int, default=5000, help='the frames start to learn')  # Reduced start learning frames
    parse.add_argument('--train-freq', type=int, default=2, help='the frequency to update the network')  # Increased update frequency
    parse.add_argument('--target-network-update-freq', type=int, default=500, help='the frequency to update the target network')  # Increased update frequency
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--display-interval', type=int, default=5, help='the display interval')  # More frequent updates on progress
    parse.add_argument('--env-type', type=str, default='atari', help='the environment type')
    parse.add_argument('--log-dir', type=str, default='logs', help='dir to save log information')
    parse.add_argument('--use-double-net', action='store_true', help='use double dqn to train the agent')
    parse.add_argument('--use-dueling', action='store_true', help='use dueling to train the agent')

    args = parse.parse_args()

    return args
