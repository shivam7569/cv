from RCNN.createFineTuneData import createFineTuneData
from RCNN.createClassifierData import createClassifierData
from RCNN.fineTune import loadBestFineTuneModel, performFineTuning
from RCNN.svm import trainSVM

##### Preparing fine tuning data #####

# createFineTuneData()

##### ************************** #####

##### Fine Tuning #####

# performFineTuning(epochs=500, debug=False, model_name="vgg16")

##### *********** #####

##### Load Model #####

# model = loadBestFineTuneModel()

##### ********** #####

##### Prepare classifier data #####

# createClassifierData()

##### *********************** #####

##### Train SVM Classifier #####

feature_model_path = "RCNN/models/checkpoints/finetune/vgg16/epoch_1_val_acc_0.7004.pt"
trainSVM(feature_model_path, epochs=25, debug=False)

##### ******************** #####

