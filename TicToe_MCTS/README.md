# Monte Carlo Tree Search  
MCTS.py is implementation of Monte Carlo Search Tree algorithm. Algorithm is made to work
with neural network. If provided with NN, instead of doing random rollouts when reaching leaf
node, does evaluation of state using NN.  
TicTacToe.py is game environment implemented by [AlexMGitHub](https://github.com/AlexMGitHub/Checkers-MCTS)
used for verifying MCTS implementation.  
# Alpha Zero
AlphaZero.py contains few important classes:  
-ResNet is neural network that will be used for training. Contains convolutional layer, residual blocks and policy and value head.  
-ResBlock is architecture of residual layer provided to ResNet class.  
-AlphaZero is training pipeline for AlphaZero algorithm. In AlphaZero.py I have tested out 
training pipeline, so file is runnable, there you can try out different NN architectures and
hyperparameters.


# Notes
Main.py was used for various testing. Most of the code is commented, and it should be divided into sections. Type of test is mentioned at the beginning of section.  
test_model.pt is saved model trained around 20 min.  
**NO PARALLELIZATION is yet implemented, everything runs on one CPU.**
