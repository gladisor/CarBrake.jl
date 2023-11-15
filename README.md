# Project Milestone, Intelligent Autonomous Systems
# Tristan Shah

![alt text](https://github.com/gladisor/CarBrake.jl/blob/main/images/car.png)

## Introduction:
In this project the overall objective has been formulated and some preliminary code modules have been written. There are still some remaining modules to be completed before experiments can be conducted. The following text will clarify the current project status.

## Problem: 
The goal of this project is to train Reinforcement Learning (RL) algorithms to control a car in a SOTA robotics simulator known as Dojo. So far the code necessary to simulate this mechanical system has been written. We currently have the ability to observe how this car moves on a flat surface in response to actuation of its wheels. This mechanism has been wrapped in an environment structure which allows convenient access to relevant data needed for RL training. 

The environment can be instantiated using the following command:

```
env = CarEnv()
```

## State:
Information about the state of the system can be observed by an agent by calling the “state” function on the environment. This will obtain the minimal representation of the mechanism at the current time point. This representation contains position, velocity, orientation, and angular velocity of the car as well as the angle and angular velocity of each wheel. This representation is in the form of a 20 element vector.

State information can be accessed through:
```
s = state(env)
```

## Action:
Actions at this point in the project are represented by 5 discrete options. The car has the ability to move forward, backwards, turn left, turn right, and idle. An additional action will be added which applies a brake pad to the wheels to slow the car. First the brake component must be added to each wheel and a spring must be attached to each pad in order for it to retract from the wheel when it is not being actuated. The action space is currently formulated as discrete however it is possible to use continuous actions instead.

The action space can be sampled through:
```
a = rand(action_space(env))
```

## Reward:
The last component needed in order to be compatible with the framework of RL is a reward signal. We will define the reward in this environment as the negative of the euclidian distance from the car to its goal location. This will incentivize the car to move towards its goal as quickly as possible. We can further enhance this project by adding a variable goal location which is supplied to the agent as state information. This would allow us to test if the RL algorithm can navigate the car to any desired location on the map. 

Reward for a particular goal is computed by calling:
```
r = reward(env)
```

## Training
Once the reward signal has been designed and implemented we will begin training of several discrete action space algorithms such as DDQN and PPO. There are a large number of algorithms implemented in the ReinforcementLearning.jl package in Julia. The performance and stability of these algorithms will be compared against one another as well as their capabilities to generalize to different goal locations.

## Additional Features
If time permits I would like to add the feature of partial observability to the control agent. We can add the ability of the car to detect randomly placed pedestrians along the path towards the goal location. Information about if a pedestrian is detected can be supplied to the agent in its state. If the car makes contact with a pedestrian the reward can be decremented by some large value. This feature would make the task significantly harder to learn and presents a more interesting problem. 

## Conclusion:
With the development of these current code modules the groundwork for an exciting project has been laid. The next step is to begin training RL agents to control the environment that has been built. If there is additional time over the next month the environment will be enhanced by adding pedestrians and sensing capabilities to the car.  