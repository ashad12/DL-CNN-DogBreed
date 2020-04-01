# DL-CNN-DogBreed
Dog Breed classifier using CNN costume model vs transfer learning

## Dataset
1- The dataset was provided and is believed to be part of [Kaggle Dog Breed dataset](https://www.kaggle.com/c/dog-breed-identification) that contains 8351 dog images

2- 13233 human face images called here as *human_files*

## Steps
The whole idea here is to make a *dog app* that is able to:

- recognize if the given picture is a human or a dog

- if a dog is detected then predict its breed.

- if a human face is detected, find a similar dog breed

### Step 1:
[Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) is used with OpenCV to find human faces in image files.
If a human face is detected the classifier outputs the location of the square
encompassing the face in terms of (x,y,w,h), where (x,y) is the coordinate of the bottom
left corner of the square and (w,h) are the width and height of the square.

NB: This step could potentially be improved by trying different face detection algorithm.

### Step 2:
vgg16 pretrained model is used to find dogs in pictures.
the model returns values 151-268 if a dog is detected.
