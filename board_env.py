import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import AECEnv



import gymnasium


from pettingzoo.utils import agent_selector, wrappers

from gymnasium.utils import EzPickle



from statistics import NormalDist

import pygame

from typing import Any , Generic, Iterable, Iterator, TypeVar
ActionType = TypeVar("ActionType")

class agent_(np.int8):
    def bucket_set(self,val=10):
        self.bucket = val
    def add(self,val=0):
        self.bucket+=val
        return self
    def sub(self,val=0):
        self.bucket-=val
        return self



class Board:
    def __init__(self,default_attack_all = True,render_= False,agent_count = 3,env_=None,use_placement_perc = False,verbose = False,max_trials = 4,max_bad_trials =4,troop_lim=51):
        # internally self.board.squares holds a flat representation of tic tac toe board
        # where an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        self.verbose = verbose
        self.env_ = env_
        self.agents = [agent_(0)] + [ agent_(i+1) for i in range(agent_count)]
        self.default_attack_all = default_attack_all
        self.render_ = render_
        self.unoccupied = agent_(0)
        self.unoccupied.bucket_set(0)
        self.reset_board()
        self.territory_count = len(self.territories)
        self.edge_count = len(self.edges)
        self.attack_dist_higher = NormalDist(mu=0, sigma=1)
        self.attack_dist_lower = NormalDist(mu=0, sigma=0.1)
        self.max_trials = max_trials
        self.max_bad_trials =max_bad_trials
        self.troop_lim = troop_lim
        self.base_action_mask =  np.zeros(self.territory_count+self.edge_count +2, dtype='int8')

        self.use_placement_perc = use_placement_perc
        #send = new_troop_count//(1/action[1])
        
        
        #[0,1,2,3,4,5,6,7,8,9]
        self.territory_map = [[0,0],[0,1],
         [0,5],[0,6],
         [2,6],[2,5],[2,4],                
         [2,2],[2,1],[2,0]]
        
        
        self.colors_territory = [(255,255,0),(128,255,0),(0,128,255),(128,0,255),
                            (255,0,0),(255,128,0),(0,255,255),(0,0,255),
                            (255,0,255),(255,0,128)]
        
    
    def reset_board(self):
        self.continents  = np.empty(4, object)
        self.continents[:] = [[0,1],[2,3],[4,5,6],[7,8,9]]
        
        self.territories = np.zeros((10,2),dtype = object)


        #0  1   --   2  3
        #   |        |
        # 9 8 7 -- 6 5 4  

        self.territories[:,0] = self.unoccupied
        
        self.edges  = np.array([[0, 1],
                                 [1, 0],
                                 [1, 2],
                                 [1, 8],
                                 [2, 1],
                                 [2, 5],
                                 [2, 3],
                                 [3, 2],
                                 [4, 5],
                                 [5, 6],
                                 [5, 2],
                                 [5, 4],
                                 [6, 7],
                                 [6, 5],
                                 [7, 8],
                                 [7, 6],
                                 [8, 9],
                                 [8, 1],
                                 [8, 7],
                                 [9, 8]],dtype='int8')
        
    
        
        self.action_list = np.zeros((len(self.territories) + len(self.edges),2))

        self.current_agent = self.agents[1]
        self.agent_counter = 0
        self.phase = 0
        self.old_phase = -1
        self.bad_trials = 0

        #[i.bucket_set(int(i)*10) for i in self.agents]
        [i.bucket_set(3) for i in self.agents]

        self.territory_changes = np.zeros(len(self.agents))
        self.set_cycle()
        self.reset_screen()

        


    def reset_screen(self):
        if self.render_:
            
            pygame.init()
            self.screen = pygame.display.set_mode((7 * 125, 3* 125))
                
    def calculated_action_mask(self, agent_sel,phase = None):
        if type(phase) == type(None):
            phase = self.phase


        
            


        
        
        
        if phase == 0:

            player_tt_cnt = sum(self.territories[:,0] == agent_sel)
            
            action_mask =  np.array([1 if ((   (i[0] == agent_sel) or ((i[0] ==0) #and (self.cycle ==0) 
                                                                       and (player_tt_cnt ==0) ) #only given a choice of empty territory in 1st cycle and can only start from one place
                                           
                                           
                                           
                                           ) and (i[1]<self.troop_lim)) else 0  
                                             for i in self.territories]
                                     +[0]*self.edge_count +[1,1], dtype='int8')
        elif phase ==1:
            action_mask =  np.array([0]*self.territory_count+[ 1 if ((agent_sel == self.territories[i[0],0]) and (self.territories[i[0],1]>1)) and (agent_sel != self.territories[i[1],0]) else 0
                                                                  for i in self.edges] +[1,1], dtype='int8')

        else:
            action_mask =  np.array([0]*self.territory_count+[ 1 if ((agent_sel == self.territories[i[0],0] == self.territories[i[1],0]) 
                                                                     and (self.territories[i[0],1]>1) 
                                                                     and (self.territories[i[1],1]<self.troop_lim)  ) else 0
                                                                  for i in self.edges] +[0,1], dtype='int8')

        

        return action_mask
    
    def take_action(self, agent_sel,action,phase = None,mask = None):
        # if spot is empty
        #action[0] is action_index
            #phase 1 : action[1] # of new troops to be added
            #phase 2 : action[1] % maximum number of troops to be sent forward if attack succeeds. #this is kept to 1 by default.

        self.reset_reward()
        
        if type(phase) == type(None):
            phase = self.phase

        action[0] = np.int8(action[0])
        
        legal,end_turn = self.check_legal(agent_sel,action,phase,mask)
        if legal and not(end_turn):


            if phase ==0:

                if self.use_placement_perc:
                    
                    self.placement(agent_sel,action,convert_percent = self.use_placement_perc,troops=self.agents[agent_sel].bucket)
                else:
                    self.placement(agent_sel,action,convert_percent = self.use_placement_perc)
            
            elif phase == 1:
                self.attack(agent_sel,action)
            else:
                self.fortify(agent_sel,action)

                end_turn = 2
                

        
        if self.render_:
            self.render()

        self.ddnt_get_a_chance_to_place()

        return legal,end_turn, self.territory_changes
        
    def check_legal(self, agent_sel,action,phase = None,mask = None):

        agent = self.agents[agent_sel]
        
        if type(phase) == type(None):
            phase = self.phase

        
        if mask == None:
            mask = self.calculated_action_mask( agent_sel,phase)

        if mask[action[0]] ==0:
            if self.verbose :
                print(1)
            self.bad_trials +=1

            _,end_=self.check_bad_trials(base_end = 0)
            
            return False, end_ 
        # only changing unoccipied or self occupied territory and within bounds
        # for phase 0 : action only adds new troops to territory

        if action[0] == 31:
            return True,2
        elif action[0] == 30:
            return True,1
        else:
            if phase ==0:

                if self.use_placement_perc:
                    if (action[1] >1) or (action[1] <0):
                        if self.verbose :
                            print(2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                    if agent.bucket == 0:
                        if self.verbose :
                            print(2.1)
                        self.bad_trials +=1
                        return False, 1
                    if (agent.bucket)*action[1] <1:
                        if self.verbose :
                            print(2.2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                
                else:    
                
                    if action[1] <1:
                        if self.verbose :
                            print(2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                    if agent.bucket == 0:
                        if self.verbose :
                            print(2.1)
                        self.bad_trials +=1
                        return False, 1
                    if agent.bucket < action[1]:
                        if self.verbose :
                            print(2.2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                    
            elif phase ==3:
                if self.use_placement_perc:
                    if (action[1] >1) or (action[1] <0):
                        if self.verbose :
                            print(2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_

                    edge = self.edges[action[0]-self.territory_count]
                    territory_a = self.territories[edge[0]]
                    
                    try:
                        if (territory_a)*action[1] <1:
                            if self.verbose :
                                print(2.2)
                            self.bad_trials +=1
                            _,end_=self.check_bad_trials(base_end = 0)
                            return False, end_
                    except:
                        print(territory_a,action[1])
                        a()
                
                else:    
                
                    if action[1] <1:
                        if self.verbose :
                            print(2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                    edge = self.edges[action[0]-self.territory_count]
                    territory_a = self.territories[edge[0]]
                    
                    if territory_a < action[1]:
                        if self.verbose :
                            print(2.2)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_                
            
            
            
            else:
                #phase 1 what percentage of troops to move forward. >=1
                if not(self.default_attack_all):
                    if action[1] >1:
                        if self.verbose :
                            print(3)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
                    if self.territories[
                                    self.edges[action[0]-self.territory_count][0]
                                    ][1]//(1/action[1]) <1          : #assuming nobody dies and (1//(1/0.999) == 0)
                        if self.verbose :
                            print(4)
                        self.bad_trials +=1
                        _,end_=self.check_bad_trials(base_end = 0)
                        return False, end_
    
            if (self.current_agent == agent) :
                if phase != self.old_phase:
                    self.agent_counter =0 # meaning change in phase
                    self.old_phase = phase
                
                elif (self.agent_counter >= self.max_trials) and (phase == self.phase):
                    if self.verbose :
                        print(4.2)
                    
                    if self.phase ==0:
                        return False, 1
                    elif self.phase ==1:
                        return False, 1
                    else:
                        return False, 2
                        
                elif (self.agent_counter < self.max_trials) and (phase == self.phase):
                    self.agent_counter +=1
                    
    
                    #elif (phase == 2):
                    #    self.agent_counter = self.max_trials
                    #    return True, 2
            
            else:
                self.set_phase(phase)
                self.set_current_agent(agent)
                self.old_phase = phase
    
        
        return True,0
            
    def check_bad_trials(self,base_end = 0):
        if self.bad_trials >= self.max_bad_trials:
            if self.verbose :
                print(0.1)
            if self.phase ==0:
                return False, 1
            elif self.phase ==1:
                return False, 1
            else:
                return False, 2
                
        return False, base_end 
        

    def placement(self,agent_sel,action,convert_percent = False,troops=[],attack=False): #action = territory index, changein troops
        territory = self.territories[action[0]]

        if convert_percent:
            
            act_ = self.convert_percent_to_troops(troops,action[1],ceil=False)
        else:
            act_ = action[1]
        
        if self.verbose :
            print('curr_agent_board',self.current_agent,agent_sel,self.agents[agent_sel], 'act',act_ , 'target',action[0],self.territories[action[0]]    )
        if not(attack) and agent_sel !=self.unoccupied:
            self.agents[agent_sel].sub(act_)
        
        #print(self.agents[agent_sel],agent_sel)
        if self.territories[action[0]][0] ==0:
            self.territory_changes[agent_sel] += 1
        
        self.territories[action[0]] = [self.agents[agent_sel],territory[1]+act_]

    def fortify(self,agent_sel,action):


        edge = self.edges[action[0]-self.territory_count]
        
        territory_a = self.territories[edge[0]]
        territory_b = self.territories[edge[1]]

        agent = self.agents[agent_sel]
        
        troop_count = territory_a[1]
        
        if self.use_placement_perc:
            send = self.convert_percent_to_troops(troop_count,action[1],ceil=False,fortify=True)
            #send = new_troop_count//(1/action[1])
        else:
            send = min(troop_count,action[1]) - 1 #send all troops but 1

        
        self.placement(agent,[edge[0],-send],convert_percent = False,attack=True)
        self.placement(agent,[edge[1],send],convert_percent = False,attack=True)


    def set_phase(self,phase=0):
        self.old_phase = self.phase
        self.phase = phase
        self.bad_trials =0
        self.agent_counter =0
        
    def set_current_agent(self,agent):
        self.current_agent = agent
    def set_cycle(self,cycle=0):
        self.cycle=cycle
    def reset_reward(self):
        self.territory_changes = np.zeros(len(self.agents))

    def convert_percent_to_troops(self,troops,percent,ceil=False, fortify=False):

        if ceil and (percent >0.9):
            send = troops -1
        else:
            send = troops//(1/(percent+0.00000001))

            if (send == troops) and (fortify) : # when trying to send all the troops during fortification... you have to leave atleast 1
                send-=1
        return send

    
    def attack(self,agent_sel,action): # this does not care about mask.... run check_legal before attacking
        self.reset_reward()
        eta = 0.0000001
        #get edge
        #get troops on a and troops on b
        if self.verbose :
            print(0)
        edge = self.edges[action[0]-self.territory_count]
        
        territory_a = self.territories[edge[0]]
        territory_b = self.territories[edge[1]]
        
        agent_a = self.agents[territory_a[0]] 
        agent_b = self.agents[territory_b[0]] 
        
        # simplify - not roll, but just check if random is less than win_chance
        diff = (territory_a[1]- territory_b[1])

        win_chance = self.attack_dist_higher.cdf( diff/(territory_b[1]+eta)  ) if diff>0 else self.attack_dist_lower.cdf( diff/(territory_b[1] +eta) )
        if self.verbose :
            print(win_chance)

        if win_chance > np.random.rand():
            #opponent_territory becomes unoccupied, and troops become zero
            #winner looses same number of troops as opponent , self troops -1 , whichever is lesser
            

            #calculate total troops left , troops to retain at pos a and troops to send at pos b 
            loss = min(territory_b[1],territory_a[1]-1)
            new_troop_count = territory_a[1] - loss

            #print(loss,new_troop_count)
            
            
            send = 0
            
            if new_troop_count>1:
                if not(self.default_attack_all):
                    send = self.convert_percent_to_troops(new_troop_count,action[1],ceil=False)
                    #send = new_troop_count//(1/action[1])
                else:
                    send = new_troop_count - 1 #send all troops but 1
            
            
            
            if (send>0): # if we want to send then send, else just vacate the territory
                self.placement(agent_a,[edge[0],-loss-send],convert_percent = False,attack=True) #-

                if self.territories[edge[1]][0] !=0: #well it would be doublecounting othervise
                    self.territory_changes[agent_a] += 1
                    
                self.placement(agent_a,[edge[1],send],convert_percent = False,attack=True) #loose equal troops always


                
                self.territory_changes[agent_b] -= 1
            else:
                self.placement(agent_a,[edge[0],-loss],convert_percent = False,attack=True) #-
                self.placement(self.unoccupied,[edge[1],-territory_b[1]],convert_percent = False,attack=True) #-
                self.territory_changes[agent_b] -= 1
                
            if (agent_b != 0):
                if self.check_territory_count_N0(agent_b ):
                    if self.verbose :
                        print('here----')
                    if type(self.env_) != type(None):
                        self.env_.terminations[agent_b] =True
                        if self.verbose :
                            print('termination --',self.env_.terminations)
                        self.territory_changes[agent_a] += 1
                        #self.territory_changes[agent_b] -= 100#50
                        
                        for ag_i,ag_term in self.env_.terminations.items():
                            if ag_term: #this means the penalization is proportional to how early you died. meaning the worse that can happen is (total agent count -1 )*-100
                                self.territory_changes[ag_i] -=100
                        
        
        else:
            loss = min(territory_b[1]-1,territory_a[1]-1)
            self.placement(agent_a,[edge[0],-loss],convert_percent = False,attack=True) #-
            self.placement(agent_b,[edge[1],-loss],convert_percent = False,attack=True) #-

    def check_territory_count_N0(self,agent):
        k = sum(self.territories[:,0] == agent)
        if self.verbose :
            print('die?',agent,k)
        return (k ==0)

    def ddnt_get_a_chance_to_place(self):
        if 0 not in self.territories[:,0]:
            for i in self.env_.possible_agents:
                if not (self.env_.terminations[i]):
                    if i not in self.territories[:,0]:
                        self.env_.terminations[i] =True
                        if self.verbose :
                            print('termination --2',self.env_.terminations)
                        #self.territory_changes[i] -= 100#50
                        for ag_i,ag_term in self.env_.terminations.items():
                            if ag_term: #this means the penalization is proportional to how early you died. meaning the worse that can happen is (total agent count -1 )*-100
                                self.territory_changes[ag_i] -=100
                        
        
    def game_status(self):
        wins = np.array( np.unique(self.territories[:,0], return_counts=True)).T
        #all territory belong to a player
        #only one player lives and others dont have anything, its phase 1 and not the 1st cycle
        if len(wins)==1:
            status = wins[0][0]

        
        elif len(wins)==2 and self.cycle !=0 and (0 in wins[:,0]):
            status = wins[:,0][wins[:,0] !=0][0]
        else:
            status = 0
        return (status),wins

    # returns:
    # -1 for no winner
    # 1 -- agent 0 wins
    # 2 -- agent 1 wins
    def check_for_winner(self):
        pass

    def check_game_over(self,agent):
        pass

    def __str__(self):
        return (self.territories)


    def render(self,last_action_txt=''):
        #pygame.time.wait(500)
        self.screen.fill((255, 255, 255))  
        width = 125
        number_font = pygame.font.SysFont( None, 40 ) 
        number_font_2 = pygame.font.SysFont( None, 20 ) 
        for i,j in zip(self.territories,self.territory_map):
            cell_left = j[1]*width
            cell_top =  j[0]*width
            pygame.draw.rect(self.screen, self.colors_territory[i[0]], (cell_left, cell_top, width, width))
            
            # make the number from grid[row][col] into an image
            if i[0]:
                number_text = 'P'+str(i[0])+'-'+str(i[1])
            else:
                number_text = 'PFree'+'-'+str(i[1])
            
            #number_text  = f'P{ (i[0] if i[0] else 'Free') }-{i[1]}'
            number_image = number_font.render( number_text, True, (0,0,0), (255,255,255) )
        
            # centre the image in the cell by calculating the margin-distance
            # Note: there is no "height", cells are square (w by w)
            margin_x = ( width-1 - number_image.get_width() ) // 2
            margin_y = ( width-1 - number_image.get_height() ) // 2
        
            # Draw the number image
            self.screen.blit( number_image, ( cell_left+2 + margin_x, cell_top+2 + margin_y ) )

        # make the number from grid[row][col] into an image
        cell_left = 3*width
        cell_top =  1*width
        
        number_text  = "|".join([f'P{i}-: bucket {i.bucket}, T:{sum(self.territories[:,0]==i)} ' for i in self.agents[1:]])
        number_image = number_font_2.render( number_text, True, (0,0,0), (255,255,255) )
    
        # centre the image in the cell by calculating the margin-distance
        # Note: there is no "height", cells are square (w by w)
        margin_x = ( width-1 - number_image.get_width() ) // 2
        margin_y = ( width-1 - number_image.get_height() ) // 2

        # Draw the number image
        self.screen.blit( number_image, ( cell_left+2 + margin_x, cell_top+2 + margin_y ) )

        # make the number from grid[row][col] into an image
        cell_left = 3*width
        cell_top =  0.8*width
        
        number_text  = f'Phase :{self.phase}, cur_Player :{self.current_agent}, turn count:{self.agent_counter}, bad turns:{self.bad_trials},act{last_action_txt} '
        number_image = number_font_2.render( number_text, True, (0,0,0), (255,255,255) )
    
        # centre the image in the cell by calculating the margin-distance
        # Note: there is no "height", cells are square (w by w)
        margin_x = ( width-1 - number_image.get_width() ) // 2
        margin_y = ( width-1 - number_image.get_height() ) // 2

        # Draw the number image
        self.screen.blit( number_image, ( cell_left+2 + margin_x, cell_top+2 + margin_y ) )
            
        
        pygame.draw.line(self.screen,(0, 0, 0),[2*width,0.5*width],[5*width,0.5*width],1)
        pygame.draw.line(self.screen,(0, 0, 0),[1.5*width,1*width],[1.5*width,2*width],1)
        pygame.draw.line(self.screen,(0, 0, 0),[5.5*width,1*width],[5.5*width,2*width],1)
        pygame.draw.line(self.screen,(0, 0, 0),[3*width,2.5*width],[4*width,2.5*width],1)

        pygame.display.flip()
        pygame.display.update() 
        





class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "risk_tiny_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 400, default_attack_all : bool| None = False,
        render_:bool|None= False,agent_count : int|None = 3,use_placement_perc : bool|None = False, add_onturn : int | None =1
    , verbose:bool|None=False , bad_mov_penalization=0.1):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.verbose = verbose
        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None
        self.phases = [0,1,2]
        self.use_placement_perc = use_placement_perc
        self.add_onturn = add_onturn
        self.last_action_txt =''
        self.bad_mov_penalization = bad_mov_penalization
        
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

        
        
        self.board = Board(default_attack_all = default_attack_all, render_=render_,agent_count = agent_count,env_=self,use_placement_perc=use_placement_perc,
                          verbose = self.verbose)

        self.agents = self.board.agents[1:] #0th is unoccupied
        self.possible_agents = self.agents[:]
        self.last_agent = self.agents[0]


        if self.use_placement_perc:
            self.action_spaces = {j: {i: ( Box(low=np.array([0,0.1]), 
                                                 high=np.array([self.board.territory_count+ self.board.edge_count+2,
                                                                         1]), 
                                     shape=(2,), dtype=(np.float16)       
                                                            )
                                      if j == 0 
                                      else Box(low=np.array([self.board.territory_count,0]), 
                                                 high=np.array([self.board.territory_count + self.board.edge_count+2,
                                                                         1]), 
                                     shape=(2,), dtype=(np.float16)       
                                                            )
                                     )
                                      
                                      
                                      
                                      for i in self.agents}
                                for j in self.phases
                             }

        else:
        
            self.action_spaces = {j: {i: ( Box(low=np.array([0,0]), 
                                                 high=np.array([self.board.territory_count+ self.board.edge_count+2,
                                                                         10]), 
                                     shape=(2,), dtype=(np.int8)       
                                                            )
                                      if j == 0 
                                      else 

                                          ( Box(low=np.array([self.board.territory_count,0]), 
                                                 high=np.array([self.board.territory_count + self.board.edge_count+2,
                                                                         10]), shape=(2,), dtype=(np.int8)  )

                                          if j ==1
                                          
                                           else 
                                          
                                          Box(low=np.array([self.board.territory_count,0]), 
                                                 high=np.array([self.board.territory_count + self.board.edge_count+2,
                                                                         1]), shape=(2,), dtype=(np.float16)    )
                                     ))
                                      
                                      
                                      
                                      for i in self.agents }
                                for j in self.phases
                             }


                                
        
        self.observation_spaces = {
            i: Dict(
                {
                    "observation":  Box(low=np.zeros(self.board.territories.shape), 
                                    high=np.ones(self.board.territories.shape)*np.array([agent_count,100]), 
                                    shape=self.board.territories.shape, dtype=agent_
                    )
                                
                    
                    
                    ,
                    "action_mask": Box(low=0, high=1, shape=(self.board.territory_count + self.board.edge_count +2,), dtype=np.int8),
                }
                
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        #self.infos = {str(i): {} for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self.cycle = 0
        self._agent_selector = agent_selector(self.agents)

        self._phase_selector = agent_selector(self.phases)


        self.handle_change(next_cycle=0,reset_cycle=1,
                      next_agent=0,reset_agent=1,
                      next_phase=0,reset_phase=1)
        

    def observe(self, agent):
        
        #observation is the same for everyone atm
        observation = np.array(self.board.territories,dtype=np.int8)

        action_mask = np.array(self._get_mask(agent),dtype=np.int8)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        
        action_mask = np.zeros(self.board.territory_count + self.board.edge_count, dtype=np.int8)

        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        if agent == self.agent_selection:
            action_mask = self.board.calculated_action_mask(agent,phase = self.phase_selection)

        return action_mask

    #def _get_obs(self):
    #    return {"agent": self._agent_location, "target": self._target_location}

    #def _get_info(self):
    #    return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def observation_space(self, agent):
        
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[self.phase_selection][agent]


    def reward_function(self,legal=True,end=0,territory_changes=[0,0,0,0],status=0,wins_=[[0,0],[0,0],[0,0],[0,0]]):

        if not(legal) and end:
            #print('subtracting')
            self.rewards[self.agents[self.agent_selection-1]] -= self.bad_mov_penalization
            self.curr_rewards[self.agents[self.agent_selection-1]] = -self.bad_mov_penalization
        elif (status != 0) and (self.all_deployed(wins_)): #game ended 
            if self.verbose :
                print('status',status)
            winner = status  # either TTT_PLAYER1_WIN or TTT_PLAYER2_WIN
            #loser = winner ^ 1  # 0 -> 1; 1 -> 0
            if self.verbose :
                print(winner)
            self.rewards[self.agents[winner-1]] += 100 #high reward forces the model to not just aqcuire territories, but also win
            self.curr_rewards[self.agents[winner-1]]=100
            for i in self.agents:
                try:
                    if i != winner:
                        self.rewards[self.agents[i-1]] -= 100
                        self.curr_rewards[self.agents[i-1]]=-100
                except Exception as e:
                    print(i,self.agents,self.rewards,winner)
                    raise Exception(e)  

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()
        else:
            for i,j in enumerate(territory_changes[1:]):
                
                #self.rewards[self.agents[i]]+=j
                self.curr_rewards[self.agents[i]]=j
                self.rewards[self.agents[i]]+=j
                

        #this is wrong ... only killed by some one elses actions
        #if (self.agent_selection not in wins_) and self.cycle>0:
        #    self.terminations[agent] =True
    #
    def handle_change(self,next_cycle=0,reset_cycle=0,
                      next_agent=0,reset_agent=0,
                      next_phase=0,reset_phase=0):
        if reset_cycle:
            self.cycle =0
            self.board.set_cycle( self.cycle)
        if next_cycle:
            self.cycle+=1
            self.board.set_cycle( self.cycle)
            
        if next_agent:
            
            if len(self._agent_selector.agent_order): # but this is not updating
                if self.verbose :
                    print('curr_agent_board',self.board.current_agent, 'curr_agent',self.agent_selection     )
                
                self.board.current_agent.add(self.add_onturn)
                self.agent_selection = self._agent_selector.next()
                while self.terminations[self.agent_selection] :
                    self.agent_selection = self._agent_selector.next()
                
                self.board.set_current_agent(self.agent_selection)
            
        if reset_agent:
            self.agent_selection = self._agent_selector.reset()
            while self.terminations[self.agent_selection] :
                    self.agent_selection = self._agent_selector.next()
            self.board.set_current_agent(self.agent_selection)
            
        if next_phase:
            self.phase_selection = self._phase_selector.next()
            
            self.board.set_phase(self.phase_selection)
        if reset_phase:
            self.phase_selection = self._phase_selector.reset()
            self.board.set_phase(self.phase_selection)
            

            
        
         

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        #print()

        self.last_action_txt = ''
        if (
            self.terminations[self.agent_selection] or ( self.agent_selection in self.kill_list ) ):
            end = 3
        elif(sum(list(self.terminations.values()) ) >=(len(self.possible_agents) -1 )):
            end =4
        else:
            legal,end,territory_changes = self.board.take_action(self.agent_selection, action)

            #if ends and illegal... give a negative rewards?
            
            status,wins_ = self.board.game_status()
            self.reward_function(legal,end,territory_changes,status,wins_)
            self.last_action_txt = f'p{self.agent_selection}, action{action}'
  


        if self.verbose :
            print('live_list',self._agent_selector.agent_order,'phase',self._phase_selector.agent_order)
        
        self.infos[self.agent_selection] = end
        
        not_removed = self.handle_post_cycle()#self.handle_terminations()
        
        
        if end==1:
            if self.verbose :
                print('1_')
            self.handle_change(next_cycle=0,reset_cycle=0,
                              next_agent=0,reset_agent=0,
                              next_phase=1,reset_phase=0)

        elif end==2:
            if self.verbose :
                print('2_')
            self.handle_change(next_cycle=1,reset_cycle=0,
                              next_agent=not_removed,reset_agent=0,
                              next_phase=0,reset_phase=1)
        elif end == 3:
            if self.verbose :
                print('3_')
            self.handle_change(next_cycle=0,reset_cycle=0,
                              next_agent=1,reset_agent=0,
                              next_phase=0,reset_phase=1)
            self.last_action_txt = f'p{self.agent_selection}, action 3_'
        elif end == 4:
            if self.verbose :
                print('3_')
            self.last_action_txt = f' P {self.agent_selection} WON!'
        
        if self.render_mode == "human":
            self.render(self.last_action_txt)

        
#######################

        

        #if self._deads_step_first():
        #self._accumulate_rewards()
        #else:
        #    self._clear_rewards()
        #    
        #if self._agent_selector.agent_order:
        #    self.agent_selection = self._agent_selector.next()

        #if self.env.frames >= self.env.max_cycles:
        #    self.terminations = dict(zip(self.agents, [True for _ in self.agents]))

        #self._cumulative_rewards[agent] = 0
        #self._accumulate_rewards()
        #self._deads_step_first()
        #self.steps += 1
##########################


        #observation = self.observe( self.agent_selection)#self._get_obs()
        #info = {observation['action_mask']}
        #observation = observation['observation']

        #return observation, reward, terminated, False, info




    

    def check_who_died(self):
        status,wins_ = self.board.game_status()
        if (self.all_deployed(wins_)):
            self.kill_list = list(set(self.agents ) - set(wins_[:,0]))
        else:
            self.kill_list = []
    
    def handle_post_cycle(self):
        not_removed = 1 
        if self._agent_selector.is_last(): #last person 2_
            self.check_who_died()
            iter_agents = self.agents[:]
            
            for agent in self.kill_list:#self.terminations:
                if ( agent in self._agent_selector.agent_order):
                    #print(agent,self.kill_list)
                    not_removed=0
                    self.terminations[agent]=True #if in killed list add to termination
                    iter_agents.remove(agent)
                    #self.agents.remove(agent)      #remove from list of alive agents

            if not(not_removed):
                self.kill_list = []
                
                self._agent_selector.reinit(iter_agents)
                #self._agent_selector.reinit(self.agents)  
                if self.verbose :
                    print('updated_order',self._agent_selector.agent_order)
        
        return 1#not_removed
        
    
    def handle_terminations(self):
        #is_last = self._agent_selector.is_last()
        not_removed = 1 
        
        iter_agents = self.agents[:]
        for agent in self.terminations:
            if (self.terminations[agent] or self.truncations[agent]) and ( agent in self._agent_selector.agent_order) :
                iter_agents.remove(agent)
                not_removed=0
                if self.verbose :
                    print('removed',agent)
        
        self.last_agent = self.agent_selection
        if not(not_removed):
            self._agent_selector.reinit(self.agents)
            if self.verbose :
                print('removed something',self._agent_selector.agent_order)
            
        
        return not_removed

    def all_deployed(self,wins):
        for i in self.agents:
            if not(self.init_deployment[i]) and ( i in wins[:,0]):
                self.init_deployment[i] =True

        return sum(self.init_deployment.values()) ==len(self.agents)


    
    def reset(self, seed=None, options=None):
        
        self.board.reset_board()
        self.curr_rewards = {i: 0 for i in self.agents}
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        #self.infos = {str(i): {} for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)

        
        self.handle_change(next_cycle=0,reset_cycle=1,
                              next_agent=0,reset_agent=1,
                              next_phase=0,reset_phase=1)
        
        self.kill_list = []

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.board.reset_screen()
            
            self.screen = self.board.screen
            pygame.display.set_caption("Tiny Risk")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((400, 700))

        self.init_deployment = {agent:False for agent in self.agents}

        

        #observation = self._get_obs()
        #info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        #return observation, info






    

    def close(self):
        pass

    def render(self,last_action_txt=''):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        self.board.render(last_action_txt)

        #if self.render_mode == "human":
        
        self.clock.tick(self.metadata["render_fps"])
        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )




def env_risk(**kwargs):
    
    kwargs['render_mode'] = kwargs['render_mode'] if kwargs['render_mode'] != "ansi" else "human"
    env = raw_env(**kwargs)
    # This wrapper is only for environments which print results to the terminal
    if kwargs['render_mode'] == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    #env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
