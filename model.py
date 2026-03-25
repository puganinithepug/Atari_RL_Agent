import torch

# a policy is governed by a function (the method of calculating next actions) where we update the weights gradually as the model learns
# creates a neural network: torch.nn.Module
class PolicyGradient(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        # takes as input number of features: print(env.observation_space)
        # states and number of actions: print(env.action_space)
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_actions)
        # initialize weights
        torch.nn.init.xavier_uniform_(self.linear.weight)
        # initialize biases
        torch.nn.init.zeros_(self.linear.bias)
        # need optimizer for training
        # specific one is a really good implementation of gradient descent
        # lr is learning rate, defined here as some small number
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        # entropy coefficient
        self.entropy_coef = 1e-5

    @staticmethod
    # note the state will be a numpy array as returned
    # convert
    def np_to_torch(x):
        # model made so that it has a 2d matrix
        t = torch.as_tensor(x, dtype=torch.float32)
        # deal with 1d case - unsqueeze
        if len(x.shape)==1:
            t = t.unsqueeze(0)
        return t

    def forward(self, states):
        return self.linear(states)
        
    def get_action(self, states):
        torch_states = self.np_to_torch(states)
        logits = self.forward(torch_states)
        # Get a proper probability distribution
        policy = self.get_policy(torch_states)
        # pick action based on probability distribution
        action = policy.sample()
        return action.item()

        
    def get_policy(self, torch_states):
        logits =self.forward(torch_states)
        return torch.distributions.Categorical(logits=logits)
        

    # want to tune reward to action, to state
    def train_step(self, states_np, actions_np, rewards_np):
        states = self.np_to_torch(states_np)
        actions = torch.as_tensor(actions_np)
        rewards = torch.as_tensor(rewards_np)

        normalize_rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)

        policy = self.get_policy(states)

        # simple policy gradient loss function
        # coming up with loss functions
        # create so it elicites a type of desired behavior
        # in ML minimze loss function 
        # if r=-1 pi = 0.9 (probab of taking that action)
        # L = -r X log(pi) minimize loss function, by pi infer pi(a|s)
        # r = -1 pi = 0.9, 0.8, 1 : L=-(-1x0.9)=0.9 L =-(-1x0.8)=0.8 L =-(-1x1)=1 with negative reward, decrease pi to minimize
        # r = 1 pi = 0.9, 0.8, 1 : L=-(1x0.9)=-0.9 L =-(1x0.8)=-0.8 L =-(1x1)=-1 with positive reward, increase pi to minimize
        # choose func to have this behavior

        log_prob = policy.log_prob(actions)

        # exploration vs exploitation
        # entropy ensures that we don't overconcentrate on a specific action - more exploring
        entropy = policy.entropy().mean()

        # loss function
        policy_loss = -(normalize_rewards*log_prob).sum()
        loss=policy_loss-self.entropy_coef*entropy

        # calculating the gradient
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
