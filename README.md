# Homeostasis_Machine

## **Step 1: Feature Extraction**

**Model Choice:** use [EfficientNet B0](https://docs.google.com/document/d/1FaLHgkX1yyqP3BC2HPN98mr7qGprXW4-m012bNmbNUU/edit?tab=t.0) (fast enough, lightweight enough, and accurate)
*   Input size is (224, 224, 3), which is (height, width, number of color channels (RGB))
*   If input image is not 224x224, it needs to be resized beore being put through the extractor
*   Layer has trainable set to false, meaning weights for that layer are frozen
    *   This avoids overfitting, saves computation during training of rest of model
    * This could be changed in the future to even further customize the learning, but the fear of overfitting is large since there is probably not a lot of data from the person compared to that in the pretrained dataset

## **Step 2: Temporal Analysis**

**Model Choice:** use LSTM-based Binary Classification Model
*   LSTM is great for sequential data
*   Type of RNN which is good at recognizing patterns
*   Number of neurons in each LSTM layer have to be enough to model necessary patterns, but small enough to run on NVIDIA Jetson Nano, thus 128 is a good balance for the first layer.
*   It is normal for the next layer to have less neurons than the first, so the next LSTM layer has 64 neurons. This is the last time step.
*   3 LSTM layers could lead to overfitting with a small dataset, so 2 layers are used
* ReLU is a simple activation function whose nonlinearity helps the model learn complex patterns
* sigmoid is a simple function that outputs a probability between 0 and 1


## **Step 3: extract_feature() Function**
*   resized.astype(np.float32): converts image (matrix of integer pixel data of size INPUT_SHAPE) to  a matrix of float32 pixel data of size INPUT_SHAPE
*   tf.expand_dims(..., axis=0): adds extra dimension at position 0 (for batch size)

## **Step 4: Main loop**
*   deque from collections used for buffer to more efficiently maintain sliding window of images
