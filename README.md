# Car vs Other Image Classification with CNN and VGG16

This project uses CNN and transfer learning with VGG16 to classify images as "car" or "other".  
The dataset folder should have subfolders for train, validation, each with classes "car" and "other".  

Set the `imgdir` variable in the script to your local dataset path.  
Install dependencies with: `pip install tensorflow numpy matplotlib`.  
Run the training script: `python code/train_cnn_vgg16.py`.  

Model weights are saved locally as `cnn.weights.h5`.  
The dataset folder is excluded from Git to avoid large uploads.  
