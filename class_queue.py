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
    def __init__(self, num_sinks=6, array_utilities=['soap', 'soap', 'towel', 'soap', 'towel', 'none'], queue_growth=.5,
                  queue_times={'soap': [10, 1], 'towel': [5, .5], 'washing': [3, .5]},
                  away_max_size = 5,
                  q_table = None):
        # Inputs:

        # num_sinks is the number of sinks

        # array_utilities contains what is available at each sink:
        # 'none' means only washing, 'soap' means washing and soap, 'towel' means washing and towels, and 'both' means everything

        # queue_growth means how many new people will get to the queue each iteration, 
        # fractional numbers means that more than one it is needed before other person appears

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
        if not all([e in ['soap', 'towel', 'none', 'both'] for e in array_utilities]):
            print("Init input invalid. Only ['soap', 'towel', 'none', 'both'] are valid")
            return
        
        # Set inputs to the obj
        self.num_sinks = num_sinks
        self.array_utilities = array_utilities
        self.queue_growth = queue_growth
        self.queue_times = queue_times
        self.away_max_size = away_max_size
        # Create or load q_table
        if q_table==None:
            self.q_table = self.get_new_q_table()
        else:
            self.q_table = q_table

        # Other initializations
        self.sinks_availability = '0'*self.num_sinks # can be 0 for 'free' or 1 for 'full'
        self.agents = []
        self.possible_needs = ['soap', 'wait', 'towel', 'washing']

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
        possible_sink_positions = [f'{i+1}' for i in range(self.num_sinks)]
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
                chosen_time_to_go == 0
                # The randomization can be improved heve, making sure that all states have a chance to be chosen

            # Generate agent
            self.add_new_agent(need=chosen_need, position=chosen_position, time=chosen_time_to_go)
        return
    
    ####################
    # get agents ready
    #####################
    def get_agents_ready_for_action(self):
        id_of_agents_ready_for_action = np.array([a.iterations_until_action for a in self.agents])
        return np.where(id_of_agents_ready_for_action==0)
    
    ##################
    # New agent
    ##################
    def add_new_agent(self, need, position):
        self.agents.append(Queue_agent(need=need, position=position))
    
    #################
    # Simulation
    #################
    def one_iteration(self, agent_id, action):
        # Passes time for all agents
        self.pass_time_all_agents()
        # Remove the finished agents
        removed_agents_count = self.remove_finished_agents()
        # Have eligible agents take actions
        
        # Generate New Agents
        # Compute Reward

        return
    
    def pass_time_all_agents(self):
        for i in range(self.agents):
            self.agents[i].iterations_until_action -= 1
    def remove_finished_agents(self):
        new_agents = []
        removed_agents_count = 0
        for i in range(self.agents):
            if not ((self.agents[i].iterations_until_action==0) and (self.agents[i].need=='towel')):
                new_agents.append(self.agents[i])
            else:
                removed_agents_count +=1
        self.agents = new_agents
        return removed_agents_count
    #################
    # Get q_table
    #################
    def get_q_table(self):
        return self.q_table
    
    def get_new_q_table(self):
        POS = [1,2,3,4,5,6,'AWAY']
        NEEDS = ['soap', 'wait', 'towel', 'washing']
        SINKS = get_binaries_array(self.num_sinks)
        QUEUE = ['LOW', 'MEDIUM', 'HIGH']
        total_states_count = len(POS)*len(NEEDS)*len(SINKS)*len(QUEUE)
        state_combinations = np.array(np.meshgrid(POS, NEEDS, SINKS,QUEUE)).T.reshape(-1,4)
        q_table_dataframe = pd.DataFrame(columns=['POS','NEEDS','SINKS','QUEUE'], data=state_combinations)
        q_table_dataframe['Q'] = 0
        return q_table_dataframe
    
    #################
    # Get state for one agent
    #################
    def get_agent_state(self, agent):
        return

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
        return int(np.random.normal(loc=mean, scale=sd))

class Queue_agent():
    ##################
    # Initialization
    ##################
    def __init__(self, need, position, time):
        # check if need is valid
        if not (need in ['soap', 'wait', 'towel', 'washing']):
            print("Invalid input. need should be one of['soap', 'wait', 'towel', 'washing']")
            return 
        if not(position in ['away', '1', '2', '3', '4', '5', '6']):
            print("Invalid input. position should be one of ['away', 1, 2, 3, 4, 5, 6]")
            return 
        
        # Atribution
        self.need = need
        self.position = position
        self.iterations_until_action = time
    
    ###################
    # Set
    ###################
    def set_need(self, need):
        self.need = need
    
    def set_position(self, position):
        self.position = position

    ###################
    # Waiting status
    ###################
    def get_waiting_status(self):
        # Return True if agent is waiting and return False if not
        # Waiting means the agent can move, but can change status
        return self.iterations_until_action>0
    
    ##################
    # Take action
    ##################
    def take_action(self, state, q_table):
        return
    


def get_binaries_array(n):
    # Return all the combinations of bits up to n bits
    A = '0'*n
    R = []
    for i in range(2**n):
        R.append("{0:b}".format(i).zfill(n))
    return R