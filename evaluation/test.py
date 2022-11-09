# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:19:21 2022

@author: 80594
"""

import torch
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from evaluation import evaluate
from class_alexnetTS112 import AlexnetTS112
from class_alexnetTS227 import AlexnetTS227
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(118)
    model_types = [112, 227]
    
    ####PARAMETER####
    model_type = model_types[0]
    
    testloader = DataLoader(GTSRB_Test_Loader(TEST_PATH='../dataset/TEST/Images', 
                                              TEST_GT_PATH = './GTSRB_Test_GT.csv', 
                                              MODEL=model_type), 
                    batch_size=50, 
                    shuffle=True, num_workers=8)
    MODELS_112 = ['BASE', 'CLASS_WEIGHTED_ONLY', 'AUGMENT_1', 'AUGMENT_2', 'CLASS_WEIGHTED_AUGMENT_1', 'LUV', 'LUV_AUGMENTED']
    MODELS_227 = ['BASE', 'CLASS_WEIGHTS_ONLY']
    
    ####PARAMETER####
    model_name = MODELS_112[3]
    if model_type == 112:
        PATH = '../models_112/' + model_name + '.pth'
        numClasses = 43
        model = AlexnetTS112(numClasses)
    else:
        PATH = '../models_227/' + model_name + '.pth'
        numClasses = 43
        model = AlexnetTS227()
    print(PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    # import your trained model 
    (testing_accuracy, testing_accuracy_classes) = evaluate(model, testloader, model_type=model_type)
    print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))

fig, ax = plt.subplots(1,1, figsize=(10,5))
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
ax.bar(range(1,44),testing_accuracy_classes)
ax.set_xlabel('Class')
ax.set_ylabel('Accuracy (%)')
# ax.set_title('Class Frequency GTSRB Training Data')
# note that the classes are imbalanced
plt.show()
print(min(testing_accuracy_classes))

#MCA
print(sum(testing_accuracy_classes)/43)


