# Get Set-Up to Write Custom Python Algorithms for Unity ML-Agents

The Python Low Level API can be used to interact directly with your Unity learning environment. As such, it can serve as the basis for developing and evaluating new learning algorithms.

The ML-Agents Toolkit Low Level API is a Python API for controlling the simulation loop of an environment or game built with Unity. This API is used by the training algorithms inside the ML-Agent Toolkit, but you can also write your own Python programs using this API.


Build your Unity project using Development Build




The `mlagents` Python package contains two components: a low level API which
allows you to interact directly with a Unity Environment (`mlagents_envs`) and
an entry point to train (`mlagents-learn`) which allows you to train agents in
Unity Environments using our implementations of reinforcement learning or
imitation learning. This document describes how to use the `mlagents_envs` API.
For information on using `mlagents-learn`, see [here](Training-ML-Agents.md).

The Python Low Level API can be used to interact directly with your Unity
learning environment. As such, it can serve as the basis for developing and
evaluating new learning algorithms.

## mlagents_envs

The ML-Agents Toolkit Low Level API is a Python API for controlling the
simulation loop of an environment or game built with Unity. This API is used by
the training algorithms inside the ML-Agent Toolkit, but you can also write your
own Python programs using this API.

The key objects in the Python API include:

- **UnityEnvironment** — the main interface between the Unity application and
  your code. Use UnityEnvironment to start and control a simulation or training
  session.
- **BehaviorName** - is a string that identifies a behavior in the simulation.
- **AgentId** - is an `int` that serves as unique identifier for Agents in the
  simulation.
- **DecisionSteps** — contains the data from Agents belonging to the same
  "Behavior" in the simulation, such as observations and rewards. Only Agents
  that requested a decision since the last call to `env.step()` are in the
  DecisionSteps object.
- **TerminalSteps** — contains the data from Agents belonging to the same
  "Behavior" in the simulation, such as observations and rewards. Only Agents
  whose episode ended since the last call to `env.step()` are in the
  TerminalSteps object.
- **BehaviorSpec** — describes the shape of the observation data inside
  DecisionSteps and TerminalSteps as well as the expected action shapes.

These classes are all defined in the
[base_env](../ml-agents-envs/mlagents_envs/base_env.py) script.

An Agent "Behavior" is a group of Agents identified by a `BehaviorName` that
share the same observations and action types (described in their
`BehaviorSpec`). You can think about Agent Behavior as a group of agents that
will share the same policy. All Agents with the same behavior have the same goal
and reward signals.

To communicate with an Agent in a Unity environment from a Python program, the
Agent in the simulation must have `Behavior Parameters` set to communicate. You
must set the `Behavior Type` to `Default` and give it a `Behavior Name`.

_Notice: Currently communication between Unity and Python takes place over an
open socket without authentication. As such, please make sure that the network
where training takes place is secure. This will be addressed in a future
release._

## Loading a Unity Environment

Python-side communication happens through `UnityEnvironment` which is located in
[`environment.py`](../ml-agents-envs/mlagents_envs/environment.py). To load a
Unity environment from a built binary file, put the file in the same directory
as `envs`. For example, if the filename of your Unity environment is `3DBall`,
in python, run:

```python
from mlagents_envs.environment import UnityEnvironment
# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="3DBall", seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()
behavior_names = env.behavior_specs.keys()
...
```
**NOTE:** Please read [Interacting with a Unity Environment](#interacting-with-a-unity-environment)
to read more about how you can interact with the Unity environment from Python.

- `file_name` is the name of the environment binary (located in the root
  directory of the python project).
- `worker_id` indicates which port to use for communication with the
  environment. For use in parallel training regimes such as A3C.
- `seed` indicates the seed to use when generating random numbers during the
  training process. In environments which are deterministic, setting the seed
  enables reproducible experimentation by ensuring that the environment and
  trainers utilize the same random seed.
- `side_channels` provides a way to exchange data with the Unity simulation that
  is not related to the reinforcement learning loop. For example: configurations
  or properties. More on them in the
  [Modifying the environment from Python](Python-API.md#modifying-the-environment-from-python)
  section.

If you want to directly interact with the Editor, you need to use
`file_name=None`, then press the **Play** button in the Editor when the message
_"Start training by pressing the Play button in the Unity Editor"_ is displayed
on the screen


```python

import numpy as np 
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs import base_env
env = UnityEnvironment(file_name="PATH TO BINARY", seed=1, side_channels=[])

env.reset()
print(f"Name of the behavior : {behavior_name}")
...
```


```python

spec = env.behavior_specs[behavior_name]

behavior_name = list(env.behavior_specs)[0]

decision_steps, terminal_steps = env.get_steps(behavior_name)

print(decision_steps.obs)
...
```

The method you will use to give the next action to agent is **`env.set_actions(behavior_name: str, action: ActionTuple)`**. The second argument `ActionTuple` is a tuple of continuous and/or discrete actions. Actions themselves are `np.arrays` with dimensions `(n_agents, continuous_size)` and `(n_agents, discrete_size)`, respectively.

```python


def GetNextAction(observations):
    #Create a 2d array of uniform random values in [-1,1] 
    continuousActionVector = 2*np.random.sample(2)-1 #Get random vector 
    continuousActionVector = continuousActionVector.reshape(1,2) #Reshape it
   
    #In this example we are not using a discrete action so 
    discreteActionVector = np.array([]).reshape(1,0);
   
    #Create an ActionTuple
    actionTuple = base_env.ActionTuple(continuousActionVector,discreteActionVector)
    
    return actionTuple

...
```


```python

numEpisodes = 10

for episode in range(numEpisodes):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0]
    # Generate an action for all agents
    
    #action = spec.create_random_action(len(decision_steps))
    randVec = 2*np.random.sample(2)-1
    action = np.array(randVec,dtype="float32") 
    
    '''ActionTuple tuple of continuous and/or discrete action.
        Actions are np.arrays with dimensions  (n_agents, continuous_size) and
        (n_agents, discrete_size), respectively.'''
 
    actionTuple = base_env.ActionTuple(randVec.reshape(1,2),np.array([]).reshape(1,0))
    
    #env.set_action_for_agent(agent_group: str, agent_id: int, action: np.array)
    
    np.shape(actionTuple._continuous)
    np.shape(actionTuple._discrete)
    
    # Set the actions
    env.set_actions(behavior_name, actionTuple)
    
    # Move the simulation forward
    env.step()
    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")
...
```


In inside the for-loop have the following while loop
```python


  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0]
    # Generate an action for all agents
...
```
    
    
```python
  
    #We generate a random action .
    randVec = 2*np.random.sample(2)-1
    action = np.array(randVec,dtype="float32") 
    
    '''ActionTuple tuple of continuous and/or discrete action.
        Actions are np.arrays with dimensions  (n_agents, continuous_size) and
        (n_agents, discrete_size), respectively.'''
 
    actionTuple = base_env.ActionTuple(randVec.reshape(1,2),np.array([]).reshape(1,0))
    
    #env.set_action_for_agent(agent_group: str, agent_id: int, action: np.array)
    
    np.shape(actionTuple._continuous)
    np.shape(actionTuple._discrete)
    
    # Set the actions
    env.set_actions(behavior_name, actionTuple)
    
    # Move the simulation forward
    env.step()
    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")
...
```

#### Custom side channels

For information on how to make custom side channels for sending additional data
types, see the documentation [here](Custom-SideChannels.md).