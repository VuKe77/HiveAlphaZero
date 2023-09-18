import os
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np


class Utils:
    """
    Utils class
    """
    def __init__(self):
        pass
    @staticmethod
    def create_model_folder(az_args):
        """
        Create new folder for model and save arguments for AZ training
        """ 
        models_directory  =os.path.join(os.getcwd(),"trained_on_server")
        if not os.path.exists(models_directory):
            os.mkdir(models_directory)
        model_saving_path = os.path.join(models_directory,f"model_{datetime.now().strftime(f'%d_%m_%H_%M_%S')}")
        os.mkdir(model_saving_path)
        #Save arguments of AlphaZero
        args_json = json.dumps(az_args,indent=4)
        with open(os.path.join(model_saving_path,"model_info.json"),"w") as f:
            f.write(args_json)
        return model_saving_path

    @staticmethod
    def _plot_loss_curve(model_saving_path):
        """
        Used for potting loss on graph. If reward is provided than plots reward also.
        """
        data_path = os.path.join(model_saving_path,"loss_data.npz")
        loss = np.load(data_path)
        policy_loss = loss['policy_loss']
        value_loss = loss['value_loss']
        reward = loss['reward']

        
        plt.figure(0)
        plt.subplot(2,1,1)
        plt.plot(policy_loss)
        plt.ylabel("Policy loss")
        plt.xlabel("epochs of training")
        plt.subplot(2,1,2)
        plt.plot(value_loss)
        plt.ylabel("value_loss")
        plt.xlabel("epochs of training")
        
        #Plot reward also
        if not (reward is None):
            plt.figure(1)
            plt.plot(reward)
            plt.ylabel("Reward")
            plt.xlabel("Number of learning iterations")
        plt.show()
