import plotly as px
import numpy as np
from functools import partial
import pandas as pd
from copy import deepcopy

class Queue():
    ##################
    # Initialization
    ##################
    def __init__(self, num_sinks=6, array_utilities=[['soap', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['washing']], queue_growth=10,
                  queue_times={'soap': [10, 1], 'towel': [5, .5], 'washing': [3, .5]},
                  away_max_size = 5, 
                  mode='collectivism', # mode can be 'collectivism' or 'egocentric'
                  collectivism_param_decay = 0.05, collectivism_param_mult = 20, 
                  egocentric_penalty = -1, egocentric_terminal_reward = 20,
                  sarsa_alpha=0.1, sarsa_gamma=0.1,
                  policy='',
                  policy_epsilon = 0.5, # The lower, the greeder
                  q_table = None):
        # Inputs:

        # num_sinks is the number of sinks

        # array_utilities contains what is available at each sink:
        # 'none' means only washing, 'soap' means washing and soap, 'towel' means washing and towels, and 'both' means everything

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
        self.collectivism_param_mult = collectivism_param_mult
        self.mode = mode
        self.egocentric_penalty = egocentric_penalty
        self.egocentric_terminal_reward = egocentric_terminal_reward
        self.sarsa_alpha = sarsa_alpha
        self.sarsa_gamma = sarsa_gamma
        self.policy_epsilon = policy_epsilon
        self.policy = policy
        # Create or load q_table
        if q_table==None:
            self.q_table = self.get_new_q_table()
        else:
            self.q_table = q_table

        # Other initializations
        self.sinks_availability = '0'*self.num_sinks # can be 0 for 'free' or 1 for 'full'
        self.agents = []
        self.possible_needs = ['soap', 'wait', 'towel', 'washing']
        self.growth_counter = 0
        self.collectivism_reward_accumulator = 0

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
    def one_iteration(self, optimize):
        # At the start: (S1,A1,R1,S2,)
        # Get elegible agents
        ids_agents_ready_for_action = self.get_agents_ready_for_action()
        agents_for_optimization = []

        # Have eligible agents take actions
        for id in ids_agents_ready_for_action:
            
            # Att state in case the last agent have moved. If not, this line should do nothing
            self.att_agents_state([id])

            action = self.agents[id].choose_action(self,self.q_table,self.policy_epsilon,self.policy)
            # print('action:',action)
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
            total_reward += egocentric_total_reward
        else:
            print('Wrong mode. Choose between collectivism and egocentric')

        # Handle finished agents
        finished_agents = self.handle_finished_agents(removed_agents)
        # Add finished agents for optimization step
        for agent in finished_agents:
            agents_for_optimization.append(deepcopy(agent))
        # SARSA
        # Need to run a optmization step on all agents in agents_for_optimization, with its (S,A,R,S,A) already right
        if optimize:
            self.multi_agents_SARSA_step(agents_for_optimization)

        return agents_for_optimization
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
                if (self.away_max_size+self.num_sinks-1)>len(self.agents):
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
        self.collectivism_reward_accumulator = self.collectivism_reward_accumulator*(1-self.collectivism_param_decay) + self.collectivism_param_mult*removed_agents*self.collectivism_param_decay
        return
    
    def att_agents_last_state(self, ids):
        for id in ids:
            self.agents[id].last_state = self.agents[id].state
            self.agents[id].last_state_idx = self.agents[id].state_idx
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
                self.single_SARSA_terminal_step(agent.last_state, agent.last_action, agent.state_idx, agent.last_state_idx)
            # Check if the agent have taken at least one action
            elif not(None in [agent.last_state, agent.last_action, agent.reward, agent.state, agent.action]):
                self.single_SARSA_step(agent.last_state, agent.last_action, agent.reward, agent.state, agent.action, agent.state_idx, agent.last_state_idx)
        return
    def single_SARSA_step(self, last_state, last_action, reward, state, action, state_idx, last_state_idx):
        # Q_S_prime_A_prime = self.get_q_value(state, action)
        # Q_S_A_idx = self.get_q_value_index(last_state, last_action)
        # self.q_table.iat[Q_S_A_idx, -1] = self.q_table.iat[Q_S_A_idx, -1] + self.sarsa_alpha*(reward + self.sarsa_gamma*Q_S_prime_A_prime - self.q_table.iat[Q_S_A_idx, -1])
        Q_S_prime_A_prime = self.q_table.iat[state_idx, -1]
        Q_S_A_idx = last_state_idx
        self.q_table.iat[Q_S_A_idx, -1] = self.q_table.iat[Q_S_A_idx, -1] + self.sarsa_alpha*(reward + self.sarsa_gamma*Q_S_prime_A_prime - self.q_table.iat[Q_S_A_idx, -1])
        return
    def single_SARSA_terminal_step(self, last_state, last_action, state_idx, last_state_idx):
        Q_S_prime_A_prime = self.egocentric_terminal_reward
        # Q_S_A_idx = self.get_q_value_index(last_state, last_action)
        # self.q_table.iat[Q_S_A_idx, -1] = self.q_table.iat[Q_S_A_idx, -1] + self.sarsa_alpha*(Q_S_prime_A_prime - self.q_table.iat[Q_S_A_idx, -1])
        Q_S_A_idx = last_state_idx
        self.q_table.iat[Q_S_A_idx, -1] = self.q_table.iat[Q_S_A_idx, -1] + self.sarsa_alpha*(Q_S_prime_A_prime - self.q_table.iat[Q_S_A_idx, -1])
        return
    #################
    # Get q_table
    #################
    def get_q_table(self):
        return self.q_table
    
    def get_new_q_table(self):
        POS = [*range(self.num_sinks), 'away']
        NEEDS = ['soap', 'wait', 'washing', 'towel']
        SINKS = get_binaries_array(self.num_sinks)
        QUEUE = ['LOW', 'MEDIUM', 'HIGH']
        ACTIONS = [*range(self.num_sinks), 'away']
        total_states_count = len(POS)*len(NEEDS)*len(SINKS)*len(QUEUE)*len(ACTIONS)
        state_combinations = np.array(np.meshgrid(POS, NEEDS, SINKS,QUEUE,ACTIONS)).T.reshape(-1,5)
        q_table_dataframe = pd.DataFrame(columns=['POS','NEEDS','SINKS','QUEUE','ACTION'], data=state_combinations)
        q_table_dataframe['Q'] = 0.0
        q_table_dataframe = q_table_dataframe.astype('category') 
        q_table_dataframe['Q'] = q_table_dataframe['Q'].astype(float)
        return q_table_dataframe
    def get_q_value_index(self, state, action):
        index = self.q_table.loc[(self.q_table['POS'] == state[0]) & (self.q_table['NEEDS']==state[1]) & (self.q_table['SINKS']==state[2]) & (self.q_table['QUEUE']==state[3]) & (self.q_table['ACTION']==action)].index.item()
        return index
    def get_q_value(self, state, action):
        q_value = self.q_table.loc[(self.q_table['POS'] == state[0]) & (self.q_table['NEEDS']== state[1]) & (self.q_table['SINKS']==state[2]) & (self.q_table['QUEUE']==state[3]) & (self.q_table['ACTION']==action), 'Q'].item()
        return q_value
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
        return len(self.agents) == self.away_max_size
    
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
        self.last_state_idx = None
        self.last_action = None
        self.reward = None
        self.state = None
        self.state_idx = None
        self.action = None
        self.id = np.random.choice(1000000)
    
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
    def choose_action(self, queue, q_table, epsilon, policy):
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
            self.state_idx = self.get_q_value_index(self.state, action)
        elif policy=='e-soft':
            # e-soft policy
            action = self.e_soft_policy(valid_actions, queue, q_table, epsilon)
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
    def e_soft_policy(self, valid_actions, queue, q_table, epsilon):
        num_actions = len(valid_actions)
        states_idxs = []
        q_s = []
        for valid_action in valid_actions:
            state = self.get_agent_state(queue.sinks_availability, queue.get_occupation())

            # q_s.append(self.get_q_value(state, valid_action, q_table))
            
            state_idx = self.get_q_value_index(state, valid_action, q_table)
            states_idxs.append(state_idx)
            q_s.append(q_table.iat[state_idx, -1])
        chances = np.zeros(len(valid_actions))
        chances[:] = epsilon/num_actions
        chances[int(np.argmax(q_s))] = 1 - epsilon + epsilon/num_actions
        action = np.random.choice(valid_actions, p=chances)

        # Here
        self.state_idx = states_idxs[np.where(np.array(valid_actions)==action)[0][0]]

        action = str(action)
        return action
    def get_q_value(self, state, action, q_table):
        q_value = q_table.loc[(q_table['POS'] == state[0]) & (q_table['NEEDS']== state[1]) & (q_table['SINKS']==state[2]) & (q_table['QUEUE']==state[3]) & (q_table['ACTION']==action), 'Q'].item()
        return q_value
    def get_q_value_index(self, state, action, q_table):
        index = q_table.loc[(q_table['POS'] == state[0]) & (q_table['NEEDS']==state[1]) & (q_table['SINKS']==state[2]) & (q_table['QUEUE']==state[3]) & (q_table['ACTION']==action)].index.item()
        return index
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



def get_binaries_array(n):
    # Return all the combinations of bits up to n bits
    A = '0'*n
    R = []
    for i in range(2**n):
        R.append("{0:b}".format(i).zfill(n))
    return R