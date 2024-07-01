import plotly as px
import numpy as np
from functools import partial
import pandas as pd
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.init as init
import copy
from torch.autograd import Variable

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Queue():
    ##################
    # Initialization
    ##################
    def __init__(self, num_sinks=6, array_utilities=[['soap', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['washing']], queue_growth=10,
                  queue_times={'soap': [10, 1], 'towel': [5, .5], 'washing': [3, .5]},
                  away_max_size = 5, 
                  mode='collectivism', # mode can be 'collectivism' or 'egocentric'
                  collectivism_param_decay = 0.05, collectivism_param_reward_scaling = 20, 
                  egocentric_penalty = -1, egocentric_terminal_reward = 20,
                  sarsa_alpha=0.1, sarsa_gamma=0.1, sarsa_beta=0.1,
                  policy_epsilon = 0.5, # The lower, the greeder
                  policy_epsilon_decay = 1,
                  q_nn = None, n_neurons=-1):
        # Inputs:

        # num_sinks is the number of sinks

        # array_utilities contains what is available at each sink:
        # 'soap' means soap, 'towel' means towels, and 'washing' means washing

        # queue_growth means how many iteration before a new person gets to the queue

        # queue_times is a obj containing the ammount of time needed for each type of action, modeled by the mean and sd of a gaussian of the time needed, 
        # time measured in iterations

        # away_max_size is the max of people awaiting away from the sinks

        # Check if inputs are right
        if len(array_utilities)!=num_sinks:
            print('Init input invalid. Number of sinks and utilities doesnt match.')
            return
        if num_sinks<1:
            print('Init input invalid. Positive number of sinks is needed.')
            return
        if not all([all([a in ['soap', 'towel', 'washing'] for a in a]) for a in array_utilities]):
            print("Init input invalid. Only ['soap', 'towel', 'none', 'both'] are valid")
            return
        
        # Set inputs to the obj
        self.num_sinks = num_sinks
        self.array_utilities = array_utilities
        self.queue_growth = queue_growth
        self.queue_times = queue_times
        self.away_max_size = away_max_size
        self.collectivism_param_decay = collectivism_param_decay
        self.collectivism_param_reward_scaling = collectivism_param_reward_scaling
        self.mode = mode
        self.egocentric_penalty = egocentric_penalty
        self.egocentric_terminal_reward = egocentric_terminal_reward
        self.sarsa_alpha = sarsa_alpha
        self.sarsa_gamma = sarsa_gamma
        self.sarsa_beta = sarsa_beta
        self.policy_epsilon = policy_epsilon
        self.policy_epsilon_decay = policy_epsilon_decay
        # Create or load q_nn
        if q_nn==None:
            self.q_nn = self.get_new_q_nn(n_neurons)
        else:
            self.q_nn = q_nn
        self.q_nn = self.q_nn.to(device)

        # Other initializations
        self.sinks_availability = '0'*self.num_sinks # can be 0 for 'free' or 1 for 'full'
        self.agents = []
        self.possible_needs = ['soap', 'wait', 'towel', 'washing']
        self.growth_counter = 0
        self.collectivism_reward_accumulator = 0
        self.optimizer = optim.SGD(self.q_nn.parameters(), lr=1)

    ###################
    # Reset state
    ###################
    def reset_state(self):
        self.sinks_availability = '0'*self.num_sinks # can be 'free' or 'full'
        self.agents = []
        self.growth_counter = 0
        self.collectivism_reward_accumulator = 0
        return
    
    ###################
    # Randomize state
    ###################
    def randomize_state(self):
        # Reset state
        self.reset_state()

        # Useful information
        possible_needs = self.possible_needs.copy()
        possible_sink_positions = [f'{i}' for i in range(self.num_sinks)]
        possible_positions = possible_sink_positions.copy()
        possible_positions.append('away')
        num_aways = 0

        # Randomize
        num_agents_to_generate = np.random.choice(range(1,self.num_sinks+self.away_max_size+1))
        
        for i in range(num_agents_to_generate):
            # Choose need
            chosen_need = np.random.choice(possible_needs)
            # Choose position
            chosen_position = np.random.choice(possible_positions)
            # Check if 'away' should be avaiable
            if chosen_position == 'away':
                num_aways += 1
            if (num_aways >= self.away_max_size) and ('away' in possible_positions):
                possible_positions.remove('away')
            if chosen_position in possible_sink_positions:
                possible_positions.remove(chosen_position)
            # Choose time
            if chosen_need=='soap':
                chosen_time_to_go = 0
            elif chosen_need=='wait':
                chosen_time_to_go = np.random.choice(self.sample_action_time('soap'))+1
            else:
                chosen_time_to_go = 0
                # The randomization can be improved heve, making sure that all states have a chance to be chosen

            # Generate agent
            self.add_new_agent(need=chosen_need, position=chosen_position, time=chosen_time_to_go)
        self.recalculate_availablity()
        return
    
    ##################
    # New agent
    ##################
    def add_new_agent(self, need, position, time):
        self.agents.append(Queue_agent(need=need, position=position, time=time))
    
    #################
    # Simulation
    #################
    def one_iteration(self, optimize, policy):
        # Decay epsilon
        self.policy_epsilon = self.policy_epsilon*self.policy_epsilon_decay

        # At the start: (S1,A1,R1,S2,)
        # Get elegible agents
        ids_agents_ready_for_action = self.get_agents_ready_for_action()
        agents_for_optimization = []

        # Have eligible agents take actions
        for id in ids_agents_ready_for_action:
            
            # Att state in case the last agent have moved. If not, this line should do nothing
            self.att_agents_state([id])

            action = self.agents[id].choose_action(self,self.q_nn,self.policy_epsilon, policy)

            # (S1,A1,R1,S2,) -> (S1,A1,R1,S2,A2)
            self.take_action(action, id)
            agents_for_optimization.append(deepcopy(self.agents[id]))
            self.recalculate_availablity()
        
        # Passes time for all agents
        self.pass_time_all_agents()

        # Remove the finished agents
        removed_agents_count, removed_agents, ids_agents_ready_for_action = self.remove_finished_agents(ids_agents_ready_for_action)
        self.recalculate_availablity()
        
        # Generate New Agents
        self.generate_new_agent_at_queue()
        self.recalculate_availablity()

        # (S1,A1,R1,S2,A2) -> (S2, A2, , S3, )
        # Att agents state
        self.att_agents_last_state(ids_agents_ready_for_action)
        self.att_agents_last_action(ids_agents_ready_for_action)
        self.att_agents_state(ids_agents_ready_for_action)

        # (S2, A2, , S3, ) -> (S2, A2, R2, S3, )
        # Compute Reward
        # Att agents reward
        total_reward = 0
        if self.mode == 'collectivism':
            # Coletivist - One reward for all agents, based on the speed of the queue
            self.att_agents_reward_collectivism(removed_agents_count, ids_agents_ready_for_action)
            total_reward = self.collectivism_reward_accumulator
        elif self.mode == 'egocentric':
            # Egocentric - Different rewards for each of the agents, based on how much time they spent there
            egocentric_total_reward = self.set_agents_egocentric_reward(ids_agents_ready_for_action)
            total_reward = egocentric_total_reward

            # Handle finished agents
            finished_agents = self.handle_finished_agents(removed_agents)
            # Add finished agents for optimization step
            for agent in finished_agents:
                agents_for_optimization.append(deepcopy(agent))
            total_reward += len(finished_agents)*self.egocentric_terminal_reward
        else:
            print('Wrong mode. Choose between collectivism and egocentric')
        
        # SARSA
        # Need to run a optmization step on all agents in agents_for_optimization, with its (S,A,R,S,A) already right
        if optimize:
            self.multi_agents_SARSA_step(agents_for_optimization)

        return total_reward
    ##########################
    # Simulation aux functions
    ##########################
    def pass_time_all_agents(self):
        # For all agents
        for i in range(len(self.agents)):
            # If they have a timer, pass time for that timer
            agent = self.agents[i]
            if self.agents[i].iterations_until_action>0:
                self.agents[i].iterations_until_action -= 1
                # print(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action,' | ID:' ,agent.id)
                # If it is the timer last iterarion, change the agent need
                if self.agents[i].iterations_until_action==0:
                    # If the agent finished with the soap, washing is needed
                    if self.agents[i].need=='wait':
                        # print(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action,' | ID:' ,agent.id)
                        self.agents[i].need='washing'
                    # If the agent finished with the washing, towel is needed
                    elif self.agents[i].need=='washing':
                        # print(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action,' | ID:' ,agent.id)
                        self.agents[i].need='towel'
                    # If the agent finished with the towel, it's done
                    elif self.agents[i].need=='towel':
                        # print(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action,' | ID:' ,agent.id)
                        self.agents[i].need='done'
                        # print('DONE!')


    def remove_finished_agents(self, ids_agents_ready_for_action):
        new_agents = []
        removed_agents = []
        removed_agents_count = 0
        new_ids_ready = []
        for i in range(len(self.agents)):
            if not (self.agents[i].need=='done'):
                new_agents.append(deepcopy(self.agents[i]))
                if i in ids_agents_ready_for_action:
                    new_ids_ready.append(len(new_agents)-1)
            else:
                removed_agents_count +=1
                removed_agents.append(deepcopy(self.agents[i]))
        self.agents = new_agents
        return removed_agents_count, removed_agents, new_ids_ready
    
    def get_agents_ready_for_action(self):
        id_of_agents_ready_for_action = np.array([(a.get_mobility_status()) for a in self.agents])
        return np.where(id_of_agents_ready_for_action==True)[0]
    
    def take_action(self, action, agent_id):
        # Check if the action taken changes the agent 'need'
        # If it's going to away, sent it there

        agent = self.agents[agent_id]
        if action=='away':
            self.agents[agent_id].set_position(action)
        else:
            # If it's going to a sink, send it there and check if the 'need' should be changed
            self.agents[agent_id].set_position(action)
            # Check if the need is met
            if self.agents[agent_id].need in self.get_sink_available_utilities(action):
                if self.agents[agent_id].need=='soap':                    
                    self.agents[agent_id].need = 'wait'
                    self.agents[agent_id].set_time(self.sample_action_time('soap'))
                elif self.agents[agent_id].need=='washing':
                    self.agents[agent_id].set_time(self.sample_action_time('washing'))
                elif self.agents[agent_id].need=='towel':
                    self.agents[agent_id].set_time(self.sample_action_time('towel'))
        # att agent action
        self.agents[agent_id].action = action

        # Recalculate sinks
        self.recalculate_availablity()

    def generate_new_agent_at_queue(self):
        self.growth_counter += 1
        # Check if growth counter is high enough
        if self.growth_counter >= self.queue_growth:
            # Check if there is space at away
            if self.away_max_size > len(self.agents):
                # Check if there is at least one space free overall
                if (self.away_max_size+self.num_sinks-2)>len(self.agents):
                    self.add_new_agent('soap', 'away', 0)
                    self.growth_counter = 0
        return
    def get_num_agents_at_away(self):
        count = 0
        for agent in self.agents:
            if agent.position == 'away':
                count += 1
        return count
    
    def att_collectivism_reward(self, removed_agents):
        # self.collectivism_reward_accumulator = self.collectivism_reward_accumulator*(self.collectivism_param_decay) + self.collectivism_param_reward_scaling*(removed_agents)*self.(1-self.collectivism_param_decay)
        self.collectivism_reward_accumulator = self.collectivism_reward_accumulator*self.collectivism_param_decay + (1-self.collectivism_param_decay)*(-len(self.agents))*(self.collectivism_param_reward_scaling)
        return
    
    def att_agents_last_state(self, ids):
        for id in ids:
            self.agents[id].last_state = self.agents[id].state
    def att_agents_last_action(self, ids):
        for id in ids:
            self.agents[id].last_action = self.agents[id].action

    def att_agents_state(self, ids):
        for id in ids:
            self.agents[id].state = self.agents[id].get_agent_state(self.sinks_availability, self.get_occupation())
        return
    def att_agents_action(self, ids):
        for id in ids:
            self.agents[id].last_action = self.agents[id].action
        return

    def set_agents_egocentric_reward(self, ids):
        total_reward = 0
        for id in ids:
            total_reward += self.egocentric_penalty
            self.agents[id].reward = self.egocentric_penalty
        
        return total_reward
    
    def att_agents_reward_collectivism(self, removed_agents_count, ids):
        self.att_collectivism_reward(removed_agents_count)
        for id in ids:
            self.agents[id].reward = self.collectivism_reward_accumulator
        return
    
    def handle_finished_agents(self, done_agents):
        for agent in done_agents:
            agent.state = 'done'
            agent.action = None
            agent.reward = self.egocentric_terminal_reward
        return done_agents
    ########################
    # SARSA
    ########################
    def multi_agents_SARSA_step(self, agents):
        for agent in agents:
            # Check if the agent is at the terminal state
            if (agent.state == 'done'):
                self.single_SARSA_terminal_step(agent.last_state, agent.last_action, agent.avg_reward)
            # Check if the agent have taken at least one action
            elif not(None in [agent.last_state, agent.last_action, agent.reward, agent.state, agent.action]):
                agent.avg_reward = self.single_SARSA_step(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action, agent.avg_reward)
        return
    def single_SARSA_step(self, last_state, last_action, reward, state, action, avg_reward):
        # Reset optimizer
        self.optimizer.zero_grad()

        # Get network inputs
        Q_S_prime_A_prime = self.q_nn(torch.tensor(state_and_action_to_network_input(state, action)).float().to(device))
        Q_S_A = self.q_nn(torch.tensor(state_and_action_to_network_input(last_state, last_action)).float().to(device))
        
        # Check avg reward
        if np.isnan(avg_reward):
            avg_reward = reward
        else:
            # Update avg reward
            avg_reward = avg_reward + self.sarsa_beta * reward

        # Calculate delta
        target = reward + Q_S_prime_A_prime
        delta = Variable(torch.tensor(target.item()), requires_grad=False) - Q_S_A

        # # Update weights
        Q_S_A.backward(delta*self.sarsa_alpha)
        self.optimizer.step()

        return avg_reward
    
    def single_SARSA_terminal_step(self, last_state, last_action, avg_reward):
        # Reset optimizer
        self.optimizer.zero_grad()

        # Get network inputs
        Q_S_A = self.q_nn(torch.tensor(state_and_action_to_network_input(last_state, last_action)).float().to(device))

        # Calculate delta
        target = self.egocentric_terminal_reward
        delta = Variable(torch.tensor(target.item()), requires_grad=False) - Q_S_A
        
        # Update weights
        Q_S_A.backward(delta*self.sarsa_alpha)
        self.optimizer.step()
        
        return
    #################
    # Get q_nn
    #################
    def get_q_nn(self):
        return self.q_nn
    
    def get_new_q_nn(self, n_neurons):
        POS = [*range(self.num_sinks), 'away']
        NEEDS = ['soap', 'wait', 'washing', 'towel']
        SINKS = get_binaries_array(self.num_sinks)
        QUEUE = [0] # Can be -1, 0 or 1
        ACTIONS = [*range(self.num_sinks), 'away']
        total_inputs_count = len(POS)+len(NEEDS)+len(SINKS[0])+len(QUEUE)+len(ACTIONS)
        new_q_nn = SLFFNN(total_inputs_count, n_neurons)
        # new_q_nn = linear_model(total_inputs_count)
        return new_q_nn
    #################
    # Get sink available utilities
    #################
    def get_sink_available_utilities(self, sink_id):
        return self.array_utilities[int(sink_id)]

    #################
    # Get queue occupation status
    #################
    def get_occupation(self):
        if len(self.agents) < self.away_max_size/3:
            return 'LOW'
        elif len(self.agents) < 2*self.away_max_size/3:
            return 'MEDIUM'
        else:
            return 'HIGH'
        
    ################
    # Sample time of action
    ################
    def sample_action_time(self, action):
        mean, sd = self.queue_times[action]
        return int(np.max([np.random.normal(loc=mean, scale=sd), 1]))

    ################
    # Get if away is full
    ################
    def is_away_full(self):
        num_agents_in_away = np.sum(np.array([agent.position for agent in self.agents])=='away')
        return num_agents_in_away == self.away_max_size
    
    def is_away_with_space_left(self):
        return len(self.agents) < self.away_max_size
    ################
    # Recalculate sinks availability
    ################
    def recalculate_availablity(self):
        agent_positions = [a.position for a in self.agents]
        if 'away' in agent_positions:
            agent_positions = list(filter(lambda a: a != 'away', agent_positions))
        new_avail = list('0'*self.num_sinks)
        for i in agent_positions:
            new_avail[int(i)] = '1'
        self.sinks_availability = ''.join(new_avail)
class Queue_agent():
    ##################
    # Initialization
    ##################
    def __init__(self, need, position, time):
        # check if need is valid
        if not (need in ['soap', 'wait', 'towel', 'washing']):
            print("Invalid input. need should be one of['soap', 'wait', 'towel', 'washing']")
            return 
        if not(position in ['away', '0', '1', '2', '3', '4', '5']):
            print("Invalid input. position should be one of ['away', 1, 2, 3, 4, 5, 6]")
            return 
        
        # Atribution
        self.need = need
        self.position = position
        self.iterations_until_action = time
        self.immobilization_states = ['washing', 'towel'] # States that, when waiting, do not let the agent take an action

        self.last_state = None
        self.last_action = None
        self.reward = None
        self.state = None
        self.action = None
        self.id = np.random.choice(1000000)
        self.avg_reward = np.nan
    
    ###################
    # Set
    ###################
    def set_need(self, need):
        self.need = need
    
    def set_position(self, position):
        self.position = position
    
    def set_time(self, time):
        self.iterations_until_action = time

    ###################
    # Waiting status
    ###################
    def get_waiting_status(self):
        # Return True if agent is waiting and return False if not
        # Waiting means the agent can move, but can change status
        return self.iterations_until_action>0
    
    ##################
    # choose action
    ##################
    def choose_action(self, queue, q_nn, epsilon, policy):
        # Choose a valid action. It should be a free sink or 'away', if it's not full
        valid_actions = []
        # Check if 'away' is not full
        if not queue.is_away_full():
            valid_actions.append('away')
        # Add free sinks
        valid_actions.extend(*np.array(np.where([not int(e) for e in list(queue.sinks_availability)]), dtype=str))
        if not (self.position in valid_actions):
            valid_actions.append(self.position)
        if policy=='random':
            # Random Policy
            action = np.random.choice(valid_actions)
        elif policy=='e-soft':
            # e-soft policy
            action = self.e_soft_policy(valid_actions, queue, q_nn, epsilon=epsilon)
        elif policy=='greedy':
            action = self.e_soft_policy(valid_actions, queue, q_nn, epsilon=0)
        else:
            print('Invalid policy')
        return action
    #################
    # Get mobility status
    #################
    def get_mobility_status(self):
        if (self.iterations_until_action>0) and (self.need in self.immobilization_states):
            return False # Cannot move
        else:
            return True # Can moves
    #################
    # e-soft
    ################
    def e_soft_policy(self, valid_actions, queue, q_nn, epsilon):
        num_actions = len(valid_actions)
        q_s = []
        for valid_action in valid_actions:
            state = self.get_agent_state(queue.sinks_availability, queue.get_occupation())
            input = state_and_action_to_network_input(state, valid_action)
            q_s.append(q_nn(torch.tensor(input).to(device).float()).cpu().detach().numpy())
        chances = np.zeros(len(valid_actions))
        chances[:] = epsilon/num_actions
        chances[int(np.argmax(q_s))] = 1 - epsilon + epsilon/num_actions
        action = np.random.choice(valid_actions, p=chances)

        action = str(action)
        return action
    def get_q_value(self, state, action, q_nn):
        input = state_and_action_to_network_input(state, action)
        value = q_nn(torch.tensor(input).float()).detach().numpy()
        return value
    ###################
    # Get state
    ###################
    def get_agent_state(self, sinks_availability, queue_occupation):
        return [self.position, self.need, sinks_availability, queue_occupation]
    ###################
    # Refresh state
    ###################
    def refresh_state(self, sinks_availability):
        self.state = self.get_agent_state(sinks_availability=sinks_availability)



###################################################################
# Other functions
###################################################################

def get_binaries_array(n):
    # Return all the combinations of bits up to n bits
    A = '0'*n
    R = []
    for i in range(2**n):
        R.append("{0:b}".format(i).zfill(n))
    return R

def plot_agents_and_rewards(window_size, agents_and_rewards_dict, mean_of_all, title=''):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    num_agents = agents_and_rewards_dict['num_agents']
    rewards = agents_and_rewards_dict['rewards']

    # Add traces
    # Agents
    num_agents_np = np.array(num_agents)
    cumsum_vec = np.cumsum(np.insert(num_agents_np, 0, 0)) 
    num_agents_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    # fig.add_trace(go.Scatter(x=np.arange(len(num_agents)), y=np.array(num_agents),
    #                     mode='lines',
    #                     name='Num agents'))

    fig.add_trace(go.Scatter(x=np.arange(window_size,window_size+len(num_agents_avg)), y=num_agents_avg,
                        mode='lines',
                        name='Num agents avg'), secondary_y=False)

    # Rewards
    rewards_np = np.array(rewards)
    cumsum_vec = np.cumsum(np.insert(rewards_np, 0, 0)) 
    rewards_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    # fig.add_trace(go.Scatter(x=np.arange(len(rewards)), y=np.array(rewards),
    #                     mode='lines',
    #                     name='Rewards'))

    fig.add_trace(go.Scatter(x=np.arange(window_size,window_size+len(rewards_avg)), y=rewards_avg,
                        mode='lines',
                        name='Rewards avg'), secondary_y=True)

    if mean_of_all:
        fig.update_layout({
            'title': f'{title}<br> Average:  Reward: {np.mean(rewards_np):.2f} | Num agents: {np.mean(num_agents_np):.2f}',
        })
    else:
        fig.update_layout({
            'title': f'{title}<br> Average in last 1k steps:  Reward: {np.mean(rewards_np[-1000:]):.2f} | Num agents: {np.mean(num_agents_np[-1000:]):.2f}',
        })
    fig.update_yaxes(title_text="Num Agents", secondary_y=False)
    fig.update_yaxes(title_text="Reward", secondary_y=True)

    fig.show()
    print(f'Correlation between number of agents and reward: {np.corrcoef(rewards_np, num_agents_np)[1,0]:.2f}')


# Single Layer Feedfoward Neural network
class SLFFNN(nn.Module):
    def __init__(self, in_shape, neurons):
        super(SLFFNN, self).__init__()
        self.fc1 = nn.Linear(in_shape, neurons)
        self.fc2 = nn.Linear(neurons, 1)        

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
class linear_model(nn.Module):
    def __init__(self, in_shape):
        super(linear_model, self).__init__()
        self.fc = nn.Linear(in_shape, 1)   

    def forward(self, x):
        x = self.fc(x)
        return x
    
# State and action to network input
def state_and_action_to_network_input(state, action):
    # Separate state
    pos = state[0]
    need = state[1]
    sinks = state[2]
    queue = state[3]
    
    # Num sinks
    num_sinks = len(sinks)
    
    # Position
    pos_input = np.zeros(num_sinks+1)
    if pos != 'away':
        if int(pos) >= num_sinks:
            return 'Error'
        pos_input[int(pos)] = 1
    else:
        pos_input[-1] = 1

    # Need
    needs = ['soap', 'wait', 'towel', 'washing']
    need_input = np.zeros(len(needs))
    need_input[needs.index(need)] = 1
    
    # Sinks
    sinks_input = [np.float64(i) for i in sinks]
    
    # Queue status
    if queue == 'LOW':
        queue_input = -1
    elif queue == 'MEDIUM':
        queue_input = 0
    elif queue == 'HIGH':
        queue_input = 1
        
    # Action
    action_input = np.zeros(num_sinks+1)
    if action != 'away':
        if int(action) >= num_sinks:
            return 'Error'
        action_input[int(action)] = 1
    else:
        action_input[-1] = 1
    
    # Concat
    nn_input = np.concatenate([pos_input, need_input, sinks_input, [queue_input], action_input])
    
    return nn_input