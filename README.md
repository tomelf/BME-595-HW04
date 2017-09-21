# BME-595 Assignment 04

The following are the speed/error VS epochs charts on the MNIST training dataset and testing dataset. According to the charts, the trending of MyImg2Num's training time and errors is the same as NnImg2Num's. However, NnImg2Num performs better on training time and errors, it might be due to a better implementation.

Experiment settings: batch-size (128), eta (0.2), epoch (5)

Training dataset: 60000 images, size of each image is 28x28

Testing dataset: 60000 images, size of each image is 28x28


## MyImg2Num
### Training time
![Training time](MyImg2Num_speed_ep-5.png)
### Training/Testing error
![Error](MyImg2Num_error_ep-5.png)

## NnImg2Num
### Training time
![Training time](NnImg2Num_speed_ep-5.png)
### Training/Testing error
![Error](NnImg2Num_error_ep-5.png)
