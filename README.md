# Cartoonification-Caption-Generation
Dataset files used -  Flickr8k_Dataset(dataset), Flickr8k_Text(description text), Flickr_8k.trainImages,Flickr_8k.devImages, Flickr_8k.testImages

Used OpenCV filters for cartoonification of images such as bilateral, greyscale and blur filters.

Here, we have used VGG (Oxford Visual Geometry Group) model, which is a pre trained model, to interpret the content of the photos.
Since it is a large model, running each photo through the network every time we want to test a new language model configuration (downstream) would be redundant.
The VGG model is loaded in Keras using VGG class, and the last layer is removed from the model as this layer is used to classify images, and VGG model is primarily used to classify images. 
Instead, the photo features are extracted using the pre-trained model and saved to a file. Before loading the images for extraction of features, the images are resized to 3 channel 224x224 pixel image.  The features are then loaded and fed to the dataset for each photo.

Next, the text file containing the descriptions for the images is loaded
We then obtain the description for each image by seperating the text in the file containing datasets line by line . The descriptions for each image is tokenized by splitting each description by whitespace. The first token for each image serves as the image identifier and filename and the remaining tokens serve as the description for each image. The image identifier and filename are then seperated by . where the first component is the image identifier and the second component is the filename. The descriptions for each image are stored in a dictionary.
We then clean the tokens present in each description for each image by converting the descriptions to lowercase and then removing numbers, trailing a's and s's, punctuation marks etc. 
We then split each description into individual words and store them into a set to remove duplicate words from description. We do this to summarise the size of vocabulary , we remove duplicate words by storing it into a set as a smaller vocabulary will result in a smaller model that will train faster. 
We now save the dictionary of image identifiers and descriptions to a file with one image identifier and description per line. 

We now train the data on all of photos and captions in the training dataset(6000 images). While training, we will monitor the performance of the model on development dataset and use that performance to decide when to save models to file.

We firstly load the training set images and extract the image identifier for each image. Then we load the descriptions and photo features for the training set images. After we convert the dictionary containing image descriptions into a list of cleaned descriptions. 

The description text will need to be encoded to numbers before it can be presented to the model as in input or compared to the model's predictions. 
The first step in encoding the data is to create a consistent mapping from words to unique integer values. Keras provides the Tokenizer class that can learn this mapping from the loaded description data.
A tokeniser is then created and fitted on the list of cleaned descriptions created, to get the internal vocabulary based on the given texts. 

Each description will be split into words as the model will be provided one word and the photo and generate the next word. Then the first two words of the description will be provided to model as input with the image to generate the next word. This is how the model will be trained.
The model is trained by transforming the data into input-output pairs of data for training the model. There are two input arrays to the model: one for photo features and one for the encoded text. There is one output for the model which is the encoded next word in the text sequence.
The input text is encoded as integers, which will be fed to a word embedding layer. The photo features will be fed directly to another part of the model. The model will output a prediction, which will be a probability distribution over all words in the vocabulary.
The output data will therefore be a one-hot encoded version of each word, representing an idealized probability distribution with 0 values at all word positions except the actual word position, which has a value of 1.

Description of the model to be used:
We will describe the model in three parts:

Photo Feature Extractor. This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.
Sequence Processor. This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
Decoder (for lack of a better name). Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction.
The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.

The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.
Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.

The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.

Fitting the model:
Now that we know how to define the model, we can fit it on the training dataset.
The model learns fast and quickly overfits the training dataset. For this reason, we will monitor the skill of the trained model on the holdout development dataset. When the skill of the model on the development dataset improves at the end of an epoch, we will save the whole model to file.
At the end of the run, we can then use the saved model with the best skill on the training dataset as our final model.
We can do this by defining a ModelCheckpoint in Keras and specifying it to monitor the minimum loss on the validation dataset and save the model to a file that has both the training and validation loss in the filename.
We can then specify the checkpoint in the call to fit() via the callbacks argument. We must also specify the development dataset in fit() via the validation_data argument.
We will only fit the model for 20 epochs, but given the amount of training data, each epoch may take 30 minutes on modern hardware.





