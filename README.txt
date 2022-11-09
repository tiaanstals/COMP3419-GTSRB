1. Folder Structure
The folder is organised as follows:

dataset
--TEST
---Images: The test images are placed in here. Note my submission does not contain the images
----00000.ppm
----...
----12629.ppm
--TRAIN: The Training images are placed here in their folders. Note my submission does not contain the images
---00000
---...
---00042
--Train.csv: This is a CSV file containing all the descriptions of the training images - used only for data exploration

evaluation
--class_alexnetTS112.py : This is a class used to build the model. This is a duplicate of the file in the directory one level back
--class_alexnetTS227.py : This is a class used to build the model. This is a duplicate of the file in the directory one level back
--evaluation.py : Evaluation function. This has been customised slightly, see comments
--test.py : Test function. This has been customised slightly, see comments
--testloader.py : Data Loader Class. This has been customised slightly, see comments
--GTSRB_Test_GT : CSV Containing Test Images and Classes


models_112 : This is a folder containing the different models output from experiments using the 112x112 network
--BASE.pth
--CLASS_WEIGHTED_ONLY.pth
--AUGMENT_1.pth
--AUGMENT_2.pth (best performing model)
--CLASS_WEIGHTED_AUGMENT_1.pth
--LUV.pth
--LUV_AUGMENTED.pth

models_227 : This is a folder containing the different models output from experiments using the 227x227 network. These models are very large (260mb) and were not included in my submission
--BASE.pth
--CLASS_WEIGHTED_ONLY.pth

class_alexnetTS112.py : This is a class used to build the model. Duplicate in the evaluation folder
class_alexnetTS227.py : This is a class used to build the model. Duplicate in the evaluation folder

model-112.ipynb: This is the notebook containing all code to build the variations of the 112 network
model-227.ipynb: This is the notebook containing all code to build the variations of the 227 network

2. Testing a model
To test the final model, all that needs to occur is you need to download the 
test images and place them in the dataset/TEST/Images folder (see 4.b). Then you can run the testing function
located in evaluation/test.py. The correct options for the final model are selected but you can change the
model being tested using the parameters. Each parameter is indicated by the ####PARAMETER#### comment. The model was
tested using a GPU but the .to(device) code should take care of that if you want to do it on a CPU.

3. To retrain a model
If you open the model-112.ipynb notebook it should be fairly self explanatory what is required
You will need to get the training images into the dataset/TRAIN folder (see 4.a). The model was
trained using a GPU but the .to(device) code should take care of that if you want to do it on a CPU. For the 227 model, it was 
trained using 2 GPUs and the code is currently setup for this

4 Downloading test and training images
4.a
To download training images and put them into the correct folder. This is to be run in the base of this directory:
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
mv GTSRB/Final_Training/Images/* dataset/TRAIN/
rm -rf GTSRB
rm -rf GTSRB_Final_Training_Images.zip

4.b
To download test imaes and put them into the correct folder. This is to be run in the base of this directory:
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_Images.zip
mv GTSRB/Final_Test/Images/* dataset/TEST/Images/
rm -rf GTSRB
rm -rf GTSRB_Final_Test_Images.zip

5. Final model
Note the final model with 97.9% accuracy is the models_112/AUGMENT_2.pth. The test.py function has been set up to test this model



