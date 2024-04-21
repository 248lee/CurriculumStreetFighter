Curriculum learning algorithm applied on other's work.
Modified from https://github.com/linyiLYi/street-fighter-ai

# Curriculum Street Fighter 2
## Introduction
This is a project that implements the curriculum learning algorithm based on resolution increasing. The algorithm adjusts the traditional reinforcement learning process, and we wish for faster convergence speed and a better resulting agent.

## Requirements
* python 3.10.12: This project is only tested on python 3.10.12. As my previous experience of using tensorflow, any version that is newer or older than 3.10.12 may cause unexpected bugs. <font color=gray>poor python...><</font>

* ```pip install stable-baselines[extra]```: This module implements the core part of reinforcement-learning-algorithms. We use the PPO in this project.

* ```pip3 install stable-retro```: This module runs the game Street Fighter 2, and wrapped it as a gym environment of OpenAI. The github page of the module is https://github.com/Farama-Foundation/stable-retro.

* **Game states and rom files**: After installing the stable-retro, we need to insert the game rom file, which is not provided by stable-retro, into the module. Download the files from https://drive.google.com/drive/folders/1ynPzMjHkEj8IQD2ne2StTk7D4ummMabH?usp=sharing, and copy all the downloaded files into the "game folder". The game folder can be obtained by running the following code:
```
import os

import retro

retro_directory = os.path.dirname(retro.__file__)
game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
print(os.path.join(retro_directory, game_dir))
```
Note that the rom file provided by the above link is only used for educational purpose. Any other usages may be ILLEGAL.

* ```pip install scipy```: This module is used for bilinear interpolation of the kernels when transferring the CNN.

* ```pip install scikit-learn```: This module is used for Kmeans operation when transferring the CNN.

* ```pip install matplotlib```: Plot gameplay screenshots.

## Functions of Each Python Files
1. ```train.py``` is the main training file.

2. ```network_structures.py``` contains the models used for the training process. There are 2 models inside, corresponding to the two stages of the curriculum learning process.

3. ```street_fighter_custom_wrapper.py``` runs the enviornment for which the agent suppose to learn. It is a wrapper of the game Street Fighter 2, and we define the observation space, action space, and reward functions in this wrapper.

4. ```test.py``` After training, we test our model here. The name of the model we want to train is specified in the code. There are also an option to choose to use RANDOM agent or TRAINED agent in the code, so make sure that ```RANDOM_ACTION=False``` before testing the trained model.

5. ```transfer.py``` is responsible for transferring a curriculum CNN from stage1 to stage2. The name of the model we want to transfer is specified in the code.

6. ```kernel_operations.py``` implements the detail operations used by ```transfer.py```, including bilinear interpolation, kmeans, and generating the old-top-layer by regression.

## How to Train
1. At stage 1, reset the variable ```STAGE=1``` in ```train.py```, and run ```python train.py```.

2. For transferring, run ```python transfer.py``` after specifying the model wishing to transfer in the python file. It will play the game first to collect data for regression. When finish transferring, it will output the transferred model ```transferred_model.zip``` in the main folder.

3. Copy the ```transferred_model.zip``` into the folder ```trained_models```, and modify the variable ```STAGE=2``` in ```train.py```, then run ```python train.py```.

## Things Modified since Last Meeting
I changed the method of downsampling the gaming screenshow from "selecting a pixel per gap" to "average pooling". In specific, I added a avgpool layer on top of the model. This causes the following effect:
1. The training speed significantly slows down (from 800 iterations/secs to 400).

2. The training result during stage1 slghtly improves. Shown in the following figure
<img src="https://i.imgur.com/qDpcx3B.png" />
