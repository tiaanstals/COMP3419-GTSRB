# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:23:37 2022
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from torchvision import transforms
import torchvision
import torch.utils.data as data
def evaluate(model, testloader, model_type=112):
    """
    Parameters
    ----------
    model : Your trained model
        DESCRIPTION: 
            It should get the img as input and generate the prediction class
            as output like follow:   output = model(img)
            output must be a probability matrix with 43 values for classification 
            For example: 
                output1 = [0.1, 0.2, 0.5, 0.1, 0, 0, 0.05, ......]
                which means output1 has the highest probability for class 2
                output2 = [0.6, 0.1, 0.05, 0.1, 0, 0, 0.05, ......]
                which means output2 has the highest probability for class 0
    testloader : The defined dataloader class
        DESCRIPTION:
            The testloader is used to get the test img and its annotation for 
            accuracy generation.
    Return:  Output the test accuracy
    -------
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_count = torch.tensor([0.0]).to(device)
    correct_count = torch.tensor([0.0]).to(device)
    class_count = np.zeros(43)
    class_sum = np.array([60, 720, 750, 450, 660, 630, 150, 450, 450, 480, 660, 420, 690, 720, 270, 210, 150, 360, 390,  60,  90,  90, 120, 150, 90, 480, 180, 60, 150, 90, 150, 270, 60, 210, 120, 390, 120, 60, 690, 90, 90, 60, 90])
    evaluate_results = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img, label, filename = data
            img = img.to(device)
            label = label.to(device)
            total_count += label.size(0)
            output = model(img)
            if model_type == 112:
                predict = torch.argmax(output[0], dim=1)
            else:
                predict = torch.argmax(output, dim=1)
            correct_count += (predict == label).sum()
            for i in range(label.size(0)):
                    if label[i] == predict[i]:
                        class_count[label[i].cpu().numpy()] = class_count[label[i].cpu().numpy()] + 1
                    else:
                        evaluate_results.append([label[i].cpu().numpy(),filename[i]])
                
    testing_accuracy = correct_count / total_count
    print("each class accuracy:\n", class_count / class_sum * 100)
    return (testing_accuracy.item(),class_count / class_sum * 100)
