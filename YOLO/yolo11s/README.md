#FAINTING RECOGNITION SYSTEM WITH TRAINED YOLO11s

##DESCRIPTION
THIS PROJECT RECREATE A FAINTING RECOGNITION SYSTEM ABLE TO CATEGORIZE TROUGH OBJECT SEGMETATION AND KEYPOINTS RECOGNITION THE POSITION OF A PERSON AS STANDING OR FAINTED. 
IT IS MEANT TO ADVANCE SECURITY FOR OPERATORS WORKING INSIDE CLEANROOMS BUT COULD BE APPLIED TO OTHER SITUATIONS.

##INSTALLATION
BEFORE RUNNING REMEMBER TO UPDATE THE FOLLOWING VARIABLES:
- MODEL_PATH = the path for the best weight of the trained model.
- VIDEO_PATH = path to the video you want to analyze.
- OUTPUT_PATH = the name you want to give to the processed video.

1. CREATE A VIRTUAL ENVIROMENT:
python -m venv venv
venv\Scripts\activate 

2. INSTALL THE LIBRARIES:
pip install -r requirements.txt

3. START THE MODEL:
py main.py

After the latter command, a debugging prewiev of the video is showing as the video is analyzed.

The video is then saved in the same folder in which the main.py file is stored.


##FEATURES
This project is based on the usage of State-of-the-Art algorithms which are the best technology at present. The images are processed by object recognition and segmentation by the model which is pre-trained to identify human keypoints. The keypoints are then utilized to estimate the position of the person: by performing in the python file the difference between the height of hips and shoulders you can determine if the person is standing or,in this case, is fainted.


##AUTHORS
MADE BY VITTORIA TAGLIABUE ON BEHALF OF IU UNIVERSITY BERLIN
