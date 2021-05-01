# BananaQ
## Project Details
A Double deep Q network (DQN) approach to the banana navigation problem.

The agent is to collect (run into) good (yellow) bananas in the environment and avoid bad (blue) bananas.
For each good banana, the agent receives a reward of +1 and for each blue one -1. The agent can take 1 of 4 actions each turn, move forward, move backward, turn left, and turn right.
The environment is considered solved if the agent wins 13 points on average.

![Banana Capture](/assets/Banana.PNG)

## Contents of this repo
* TrainModel.ipynb: a notebook that trains a model and saves model to model.pt
* RunModel.ipynb: a notebook that loads and runs a trained model 
* TestModel.ipynb: a notebook that loads and test the model for 100 iterations and prints the graph and average score
* model.py: a 3 hidden layer triangle shaped model using ReLU activation functions built using py torch  
* dq_agent.py: the DQN agent that runs using model.py, which can be adapted to use any similar learner. The network as built will learn to solve this challenge in around 530 episodes.
* Report.pdf

## Installing the Repo
* Install Anaconda and run the Jupyter notebook from there.

`conda install pytorch=0.4.0 -c pytorch`
* pip install dependencies under python folder. 

`!pip -q install ./python`
* Downlaod Unity Enviorment and replace contents of Banana folder (I have included the windows 64 bit version):
  * [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  * [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  * [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  * [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  
  
## Run
* Run TrainModel.ipynb to train a new model
* Run RunModel.ipynb to see the agent play the game.
