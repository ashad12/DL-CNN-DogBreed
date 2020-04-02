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

### Step 3:
a costume model is proposed to detect dg breed. The procedure was as follows:
1- I had initially started with only convolutional layers and was limited to maximum 64 channels since greater values would pop memory error. However, the best accuracy I achieved was about 2%. This increased to 5% by implementing image augmentation. Furthermore, I attempted to eliminate maxpooling layers and instead use additional convolutional layers that would half the spatial size of the image (kernel_size=4, stride=2, padding=1). However, the model diverged.

2- I mixed up skip-connections and batch normalization methods and increased my accuracy to 9%.

3- Interestingly, by removing the skip connections and only using batch normalizations after each convolutional layers, I was able to increase the channels up to 128 and achieve 16% accuracy. It is believed using larger channels may further improve the accuracy.

Thereby, the architecture of the final model is as follows:

- 4 convolutional layers: the output channel doubles consecutively after each layer, starting form 16 to 128.
- 3 batch normalization layers each of which is applied to the output of the convolutional layer except for the last convolutional layer.
- 4 maxpooling layers after each convolutional layer
- 3 fully connected layers to flatten data and obtain 133 nodes as the classes of the dog breeds. Dropout layers are added after the first and second fully connected later with the probability of 0.25 to avoid overfitting and better generalizing the model.
- I used relu as the activation function for each layer

**NB:** There was GPU usage limitation both time-wise and memory-wise so deeper and more complex model
was not practical

**Optimization Parameters and Training**:
SGD was used with lr=0.01, smaller lr required longer training and larger lr would
diverge.
Cross Entropy loss was used for this classification problem.
Training function is set to 100 epoch training, though, an early_stop parameter is
also included to save the limited GPU hours in the workspace, which would terminate the loop if the validation loss is rising continuously. This parameter receives a value between 0 and 1 which is the ratio of the amount of the validation loss increase to the minimum recorded validation loss. If the validation loss becomes greater than (1 + early_stop)* valid_loss_min for 3 consecutive iteration, the training loop would break.
