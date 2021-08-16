PYTHON=python3
# arguments: buffer_size model_buffer_size batch_size train_freq gradient_steps

#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 1 0 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 2 0 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 3 0 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 5 0 1 1 1
CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 10 0 1 1 1
CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 20 0 1 1 1
CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 30 0 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 4 1 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 3 2 1 1 1
#CUDA_VISIBLE_DEVICES=0 $PYTHON minimum_working_example_model_based_DQN.py 2 3 1 1 1
