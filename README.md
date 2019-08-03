# Fast-Frontal-Face-and-Eye-Detection-using-Viola-Jones-Object-Detection
This project was submitted for the Assignment component of the UoM ICS3206 - Machine Learning, Expert Systems and Fuzzy Logic unit. A haarcascade was trained in order to detect faces and eyes within images in real time. 

Please note that the 'evaluation', 'neg', 'neg_raw' and 'pos' folders and the 'bg.txt', 'info.txt' and 'faces.vec' are only a sample of the total data, and only a sample was uploaded due to size limitations.\
The entire project can be accessed on Google Drive by following this link: 
https://drive.google.com/drive/folders/1RVbeVsix6Uer_D9wIY0FM5Yp4dwQuR2S?usp=sharing

The images used are all .jpg

To train the cascade:\
Positive training images are in the 'pos' folder, they should be cropped to a person's face, and be 178x218.\
To use a different size, change line 42 of TrainingHaar.py

There are a number of sub-folders in the 'neg_raw' folder, each folder contains images from their respective dataset.\
Raw negative images can be placed in any of those folders. These can be of any size, as long as they don't contain any faces.

When running TrainingHaar.py, this will normalise all the negative images to grayscale 250x250 and place them in the 'neg' folder.\
It will also create the bg.txt and info.txt files. Please make sure that these have Unix line endings for opencv to run correctly.\
(All the required packages to run the python files are included in 'python_package_requirements.txt', however these should be installed by simply running 'pip install opencv-python' on python 3.6)\
(Also make sure that any old versions of info.txt and bg.txt are deleted before running the python script, as it appends data, not overwrites it)

The vector file, faces.vec, can be created by running:\
opencv_createsamples -info info.txt -num 202599 -w 24 -h 24 -vec faces.vec\
202599 is the number of images in the pos folder, 24x24 are the dimensions of the images in the vec file, these must match the dimensions used in the traincascade command\
(This command should overwrite any old versions of faces.vec, however it is safer to just delete them before running this command)

The cascade can be trained by running:\
opencv_traincascade -data trained_cascade -vec faces.vec -bg bg.txt -numPos 150000 -numNeg 75000 -numStages 15 -w 24 -h 24\
'trained_cascade' is the folder name where the generated cascade will be stored\
(the 'trained_cascade' folder should be created and left empty, as this command does not create it, and crashes when it does not find the folder)

To run the cascade:\
Run 'PreTrainedCascade.py', you will be asked to either write 'img', and the cascade will detect faces and eyes in a random image from the 'evaluation' directory\
or write 'cam' to detect faces and eyes on a stream from the webcam\
or enter the path to an image to detect faces and eyes in that\
or 'quit' to stop execution.

This python script currently uses a pretrained cascade, to change this simply go to line 5 and specify a different cascade, such as one from the 'mycascades' folder

To evaluate all the cascades:\
Run the 'Evaluation.py' script which will test each cascade against each image in the evaluation directory.
