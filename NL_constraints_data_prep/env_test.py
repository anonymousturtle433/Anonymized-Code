import gym
# import multiagent
# from multiagent.environment import MultiAgentEnv
# import multiagent.scenarios as scenarios
import gym_risk
import logging
import random
import time
from curses import wrapper
import curses
LOG = logging.getLogger("pyrisk")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nocurses", dest="curses", action="store_false", default=True, help="Disable the ncurses map display")
parser.add_argument("--nocolor", dest="color", action="store_false", default=True, help="Display the map without colors")
parser.add_argument("-l", "--log", action="store_true", default=False, help="Write game events to a logfile")
parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay in seconds after each action is displayed")
parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
parser.add_argument("-w", "--wait", action="store_true", default=False, help="Pause and wait for a keypress after each action")
# parser.add_argument("players", nargs="+", help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
parser.add_argument("--deal", action="store_true", default=False, help="Deal territories rather than letting players choose")
parser.add_argument("--state", required = True, type=int, default=0, help="Initial state representation")

args = parser.parse_args()

logging.basicConfig(filename="pyrisk.log", filemode="w")
LOG.setLevel(logging.INFO)


#this function appears to be completely irrelevant to the functioning of the environment
def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

kwargs = dict(curses=args.curses, color=args.color, delay=args.delay,
              wait=args.wait, deal=args.deal, state=args.state)

def run(stdscr, **kwargs):
    
    env = gym.make('Risk-v0')
    env.reset(stdscr, **kwargs)
    

    #[action to take, start location (given by action map), end location (irrelevant for drafting, ]
    # draft_actions = [[0,2,12,2],[0,3,12,2],[0,4,12,2],[0,5,12,2],[0,6,12,2],[0,7,12,2],[0,8,12,2]]
    #valid control yellow
    draft_actions = [[0,0,1,3], [0,1,1,3], [0,2,1,4], [0,3,1,4]]
    #valid don't control yellow
    draft_actions = [[0,0,1,3], [0,1,1,3], [0,2,1,8]]
    
    #draft actions that should result in a loss:
    #result - all lossses
    #owned by opponent
    # draft_actions = [[0,4,1,3]]
    # draft_actions = [[0,8,1,3]]
    #results in more than 14 troops
    # draft_actions = [[0,0,1,3], [0,0,1,12]]
    # draft_actions = [[0,0,1,25]]
    #


    # reinforce_actions = [[1,2,12,2],[1,3,12,2],[1,4,12,2],[1,5,12,2],[1,6,12,2],[1,7,12,2],[1,8,12,2],[1,9,12,2],[1,10,12,2],[1,11,12,2],[1,11,12,2],[1,13,12,2],[1,14,12,2],[1,15,12,2],[1,16,12,2]]
    
    # reinforce_actions = [[1,0,1,1], [1,1,1,1], [1,2,1,1], [1,3,1,1]]
    #reinforce actions that should result in a lost
    #wrong action
    # reinforce_actions = [[0,4,1,1]]
    #owned by opponent
    # reinforce_actions = [[1,4,1,1]]
    # reinforce_actions = [[1,8,1,1]]
    #owned by no one
    # reinforce_actions = [[1,10,1,1]]
    #more than troops remaining
    # reinforce_actions = [[1,0,1,4]]

    #correct reinforcement
    reinforce_actions = [[1,0,1,1]]

    attack_actions = [[2, 2, 3, 2], [2, 0, 3, 2], [2, 1, 2, 2], [2, 3, 8, 2], [2, 4, 5, 2], [2, 5, 6, 2], [2, 4, 7, 2], [2, 6, 16, 2], [2, 8, 9, 2], [2, 9, 10, 2], [2, 10, 11, 2], [3, 2, 3, 1], [3, 0, 3, 1], [3, 1, 2, 1], [3, 3, 8, 1], [3, 4, 5, 1]]
    phase = 'drafting'
    phases = ['drafting', 'reinforce', 'attack']
    phase_index = 0
    j = 0
    for i in range(10): #originally 1000
        print(f'agent action: {i}')
        
        # act = env.action_space.sample()
        # if phases[j%3] == 'drafting':
        #     act = draft_actions[i]
        # elif phases[j%3] == 'reinforce':
        #     act = random.choice(reinforce_actions)
        # elif phases[j%3] == 'attack':
        #     act = random.choice(attack_actions)

        if phase == 'drafting':
            act = draft_actions[i]
        elif phase == 'reinforce':
            act = random.choice(reinforce_actions)
        elif phase == 'attack':
            act = random.choice(attack_actions)

        LOG.info(act)
        (p, phase_done, state), reward, done, _ = env.step(act) # take a random action
        LOG.info(state)
        LOG.info((p, phase_done, reward, done))
        
        if done:
            print('Game over')
            LOG.info(p)
            break

        if phase_done and phase == 'drafting':
            phase_index = 1
            phase = phases[phase_index]
        elif phase_done and phase == 'reinforce':
            phase_index = 2
            phase = phases[phase_index]
        elif phase_done and phase == 'attack':
            phase_index = 1
            phase = phases[phase_index]

        # env.render(state)
        env.render()

        # if phase_done: 
        #     j += 1
        # time.sleep(2)
    env.close()


# env = gym.make('Risk-v0')
# env.reset(None, **kwargs) 
   

# for _ in range(100):
#     env.render()
#     print("inside")
#     act = env.action_space.sample()
#     print(act)
#     print(env.step(act)) # take a random action
# env.close()

if args.curses:
    wrapper(run, **kwargs)
    # assert False
else:
    run(None, **kwargs)