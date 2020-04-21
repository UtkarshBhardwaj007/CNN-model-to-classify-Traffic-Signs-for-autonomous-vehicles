# CNN-model-to-classify-Traffic-Signs-for-autonomous-vehicles
The CNN model is built for the purpose of classification of traffic signs detected by autonomous vehicles into 43 classes.
The model takes in images as input and classifies them into 43 different class labels. Since a sigmoid function is used in the output layer, the model returns the probability of an input being of each of the 43 categories.
The model architecture can be viewed by execution the print(model.summary()) statement in the trafficSignsClassifier.py file.
All the parameters that were used for training are specified in the parameters section of the code. To train the model on your local device you will need to modify the paths.

There are about 380000 trainable parameters due to the model architecture. The training was performed on 15 epochs and the graph of training accuracy and validation accuracy is named as accuracy.png in the repo.
The graph of training loss and validation loss vs epochs is named as loss.pnd in the repository.

The model was trained on about 26000 images with a 0.2 split for validation data. The testing set consisted of about 12000 images and the accuracy achieved on it was 98%.
The classification report can be found in the repository by viewing the classification_report.png file.

Since the confusion matrix was a 43x43 matrix, I plotted a heatmap using seaborn's heatmap for better visualisation.
The actual names and the 43 associated labels are given in signnames.csv.

It can be noticed in the heatmap that the diagonal cells contain most of the elements but they don't seem to have even nearly the same number of elements. That is because the dataset was pretty uneven (not the same number of test images for all classes). We can visualise this by plotting a histogram of the testing data based on class. I leave this task to the readers.

I also tried to demonstrate two different methods of importing data images through flow_from_directory and using cv2.imread after specifying the path with os.listdir().

I have provided instructions for saving your model within the code itself.

If you wish to load my model to test it or compare or for any other reason, you can do so as I have provided the h5  file of my model in the repository itself. The instructions to load any model are simple and are given in the code itself.

The dataset I used is public and can be downloaded from - https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
