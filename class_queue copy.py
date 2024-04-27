try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    import gym
    from gym import spaces
import plotly as px
import numpy as np
from functools import partial
import pandas as pd

class Queue():
    ##################
    # Initialization
    ##################
    def __init__(self, num_sinks=6, array_utilities=[['soap', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['washing']], queue_growth=10,
                  queue_times={'soap': [10, 1], 'towel': [5, .5], 'washing': [3, .5]},
                  away_max_size = 5, collectivism_param_decay = 0.05, selfcollectivism_param_mult = 10,
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
        self.selfcollectivism_param_mult = selfcollectivism_param_mult
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
    def one_iteration(self, agent_id, action):
        # Att agents last
        self.att_agents_last()
        # Passes time for all agents
        self.pass_time_all_agents()
        # Remove the finished agents
        removed_agents_count = self.remove_finished_agents()
        # Have eligible agents take actions
        ids_agents_ready_for_action = self.get_agents_ready_for_action()
        for id in ids_agents_ready_for_action:
            action = self.agents[id].choose_action()
            self.take_action(action, id)
            self.recalculate_availablity()
        # Generate New Agents
        self.generate_new_agent_at_queue()
        # Att agents state
        self.att_agents_state(self.sinks_availability)
        # Att agents actions
        self.att_agents_actions()
        # Compute Reward
        # Coletivist - One reward for all agents, based on the speed of the queue
        collectivism_reward = self.att_collectivism_reward(removed_agents_count)
        # Egocentric - Different rewards for each of the agents, based on how much time they spent there
        # egocentric_reward = self.get_egocentric_reward()
        return
    
    def pass_time_all_agents(self):
        # For all agents
        for i in range(self.agents):
            # If they have a timer, pass time for that timer
            if self.agents[i].iterations_until_action>0:
                self.agents[i].iterations_until_action -= 1
                # If it is the timer last iterarion, change the agent need
                if self.agents[i].iterations_until_action==0:
                    # If the agent finished with the soap, washing is needed
                    if self.agents[i].need=='wait':
                        self.agents[i].need='washing'
                    # If the agent finished with the washing, towel is needed
                    elif self.agents[i].need=='washing':
                        self.agents[i].need='towel'
                    # If the agent finished with the towel, it's done
                    elif self.agents[i].need=='towel':
                        self.agents[i].need='done'

    def remove_finished_agents(self):
        new_agents = []
        removed_agents_count = 0
        for i in range(self.agents):
            if not (self.agents[i].need=='done'):
                new_agents.append(self.agents[i])
            else:
                removed_agents_count +=1
        self.agents = new_agents
        return removed_agents_count
    
    def get_agents_ready_for_action(self):
        id_of_agents_ready_for_action = np.array([(a.get_mobility_status()) for a in self.agents])
        return np.where(id_of_agents_ready_for_action==True)
    
    def take_action(self, action, agent_id):
        # Check if the action taken changes the agent 'need'
        # If it's going to away, sent it there
        if action=='away':
            self.agents[agent_id].set_position = action
        else:
            # If it's going to a sink, send it there and check if the 'need' should be changed
            self.agents[agent_id].set_position = action
            # Check if the need is met
            if self.agents[agent_id].need in self.get_sink_available_utilities(action):
                if self.agents[agent_id].need=='soap':
                    self.agents[agent_id].need = 'wait'
                    self.agents[agent_id].set_time(self.sample_action_time('soap'))
                elif self.agents[agent_id].need=='washing':
                    self.agents[agent_id].set_time(self.sample_action_time('washing'))
                elif self.agents[agent_id].need=='towel':
                    self.agents[agent_id].set_time(self.sample_action_time('towel'))
        self.agents[agent_id].action = action

        # Recalculate sinks
        self.recalculate_availablity()

    def generate_new_agent_at_queue(self):
        self.growth_counter += 1
        if self.growth_counter >= self.queue_growth:
            if self.away_max_size > len(self.agents):
                self.add_new_agent('soap', 'away')
        return
    
    def att_collectivism_reward(self, removed_agents):
        self.collectivism_reward_accumulator = self.collectivism_reward_accumulator*(1-self.collectivism_param_decay) + self.collectivism_param_mult*removed_agents*self.collectivism_param_decay
        return self.collectivism_reward_accumulator
    
    def att_agents_last(self):
        for id in range(len(self.agents)):
            self.agents[id].last_state = self.agents[id].state
            self.agents[id].last_action = self.agents[id].action
            self.agents[id].last_reward = self.agents[id].reward

    def att_agents_state(self):
        for id in range(len(self.agents)):
            self.agents[id].state = self.agents[id].get_agent_state(self.sinks_availability)
        return
    #################
    # Get q_table
    #################
    def get_q_table(self):
        return self.q_table
    
    def get_new_q_table(self):
        POS = [1,2,3,4,5,6,'AWAY']
        NEEDS = ['soap', 'wait', 'washing', 'towel']
        SINKS = get_binaries_array(self.num_sinks)
        QUEUE = ['LOW', 'MEDIUM', 'HIGH']
        total_states_count = len(POS)*len(NEEDS)*len(SINKS)*len(QUEUE)
        state_combinations = np.array(np.meshgrid(POS, NEEDS, SINKS,QUEUE)).T.reshape(-1,4)
        q_table_dataframe = pd.DataFrame(columns=['POS','NEEDS','SINKS','QUEUE'], data=state_combinations)
        q_table_dataframe['Q'] = 0
        return q_table_dataframe
    
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
        return int(np.max(np.random.normal(loc=mean, scale=sd), 0))

    ################
    # Get if away is full
    ################
    def is_away_full(self):
        return len(self.agents) == self.away_max_size
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
        self.last_reward = None
        self.state = None
        self.action = None
        self.reward = None
    
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
    def choose_action(self, queue, q_table):
        # Choose a valid action. It should be a free sink or 'away', if it's not full
        valid_actions = []
        # Check if 'away' is not full
        if not queue.is_away_full():
            valid_actions.append('away')
        # Add free sinks
        valid_actions.extend(*np.where([not int(e) for e in list(queue.sinks_availability)]))

        # Random Policy
        action = np.random.choice(valid_actions)
        return action
    #################
    # Get mobility status
    #################
    def get_mobility_status(self):
        if (self.iterations_until_action>0) and (self.need==self.immobilization_states):
            return False # Cannot move
        else:
            return True # Can moves
    #################
    # Pass time
    ################
    def pass_time(self):
        # If they have a timer, pass time for that timer
        if self.iterations_until_action>0:
            self.iterations_until_action -= 1
            # If it is the timer last iterarion, change the agent need
            if self.iterations_until_action==0:
                # If the agent finished with the soap, washing is needed
                if self.need=='wait':
                    self.need='washing'
                # If the agent finished with the washing, towel is needed
                elif self.need=='washing':
                    self.need='towel'
                # If the agent finished with the towel, it's done
                elif self.need=='towel':
                    self.need='done'
    ###################
    # Get state
    ###################
    def get_agent_state(self, sinks_availability):
        return [self.position, self.need, sinks_availability]



def get_binaries_array(n):
    # Return all the combinations of bits up to n bits
    A = '0'*n
    R = []
    for i in range(2**n):
        R.append("{0:b}".format(i).zfill(n))
    return R