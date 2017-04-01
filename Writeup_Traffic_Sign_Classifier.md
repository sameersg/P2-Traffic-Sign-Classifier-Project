#**Traffic Sign Recognition** 

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Visualization"
[image2]: ./examples/org.png "Original"
[image3]: ./examples/grey.png "Grayscaling"
[image4]: ./examples/norm.png "Normalization"
[image5]: ./examples/20.jpg "Traffic Sign 1"
[image6]: ./examples/60.jpg "Traffic Sign 2"
[image7]: ./examples/fuss.jpg "Traffic Sign 3"
[image8]: ./examples/gefahr.jpg "Traffic Sign 4"
[image9]: ./examples/geradeoderrecht.jpg "Traffic Sign 5"
[image10]: ./examples/kein_speed.jpg "Traffic Sign 6"
[image11]: ./examples/keineinfahrt.jpg "Traffic Sign 7"
[image12]: ./examples/kreis.jpg "Traffic Sign 8"
[image13]: ./examples/rechts.jpg "Traffic Sign 9"
[image14]: ./examples/stop.jpg "Traffic Sign 10"
[image15]: ./examples/uneben.jpg "Traffic Sign 11"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 😉 and here is a link to my [project code](hhttps://github.com/sameersg/P2-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the **second code cell** of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of test set is **12630**
* The shape of a traffic sign image is **(34799, 32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the **fourth code cell **of the IPython notebook.  

Here is a visualization of the date set. It shows a bar chart of how much data is available for the 43 Sign classes. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the **fifth code cell** of the IPython notebook.

As a first step, I decided to convert the images to grayscale because color information is not necesserary to identify a german sign. And second i normalize the image data with Min-Max scaling to a range of [0.1, 0.9]. After that i reshape the image for CNN. At last i shuffle the data. 

Here is an example of a traffic sign image before, after grayscaling and after normalize.

![alt text][image2] ![alt text][image3] ![alt text][image4] 


####2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **seventh cell** of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16 |
| RELU					|	
| Max pooling	      	| 2x2 stride,  output 5x5x16						|
| Flatten				| Output 400
| Fully connected 1		| Output 120       									|
| RELU					|
| Dropout					| 
| Fully connected 2		| Output 84 
| RELU					|
| Fully connected 3		| Output 43

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the **eigth and the nineth cell**  of the ipython notebook. 

To train the model, I used EPOCHS = 30, BATCH_SIZE = 128 and rate = 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the **eleventh cell** of the Ipython notebook.

My final model results were:
* training set accuracy of **0.995**
* validation set accuracy of **0.923**
* test set accuracy of **0.923**

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

	LeNet architecture because i understood it from the lectures.
  
* What were some problems with the initial architecture?

	The accuracy was with the RGB images very low, after greyscaling and normalising and adding Dopouts it got better.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

	The model was overfitting, adding dropouts helped to get a good  validation accuracy of 0.946.



* Which parameters were tuned? How were they adjusted and why?
	
	Epochs where tuned to 30 for a good result. I tried once a epoch of 100 this gave me a training set accuracy of 1.000 and validation set accuracy of 0.951, but it was very time consuming. Modifing batch_size and rate were not improving my result so i stick with 128 and 0.001.
	

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

	Dropouts helped the model to  learn better a get a validation above 0.93.

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eleven German traffic signs that I found on the web:

![alt text][image5]
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9] 
![alt text][image10] 
![alt text][image11] 
![alt text][image12] 
![alt text][image13] 
![alt text][image14] 
![alt text][image15]

The sixt image (End of all speed and passing limits) might be difficult to classify because it ist very similar to other signs like End of no passing sign. So this was not gettting reconized every time. But i got a overall accuracy of **0.909.**


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the **fifteenth cell** of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h) 	| Speed limit (20km/h)
| Speed limit (60km/h) 	| Speed limit (60km/h)
| Pedestrians				| Pedestrians		
| General caution      	| General caution
| Go straight or right	| Go straight or right
| End of all speed and passing limits	| End of no passing
| No entry				| No entry	
| Roundabout mandatory	| Roundabout mandatory
| Turn right ahead		| Turn right ahead
| Stop						| Stop
| Bumpy road				| Bumpy road
			
      							


The model was able to correctly guess 10 of the 11 traffic signs, which gives an accuracy of 90%. This ist better then the test set that had a accuracy of 92%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the **sixteenth cell** of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Speed limit (20km/h) 
| .08     				| Speed limit (70km/h)
| .00001					| Speed limit (30km/h)
| .00000002	      			| Stop	
| .000000007					    | Speed limit (60km/h)
    							


For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Speed limit (60km/h) 
| .005     				| Speed limit (30km/h)
| .000005					| Speed limit (50km/h)
| .00000000002	      			| Stop	
| .000000000000000008					    | Speed limit (20km/h)




For the image that not got recognized correctly


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .50         			| End of no passing							 
| .47     				| End of all speed and passing limits|
| .01					| End of speed limit (80km/h)
| .00008	      			| Bicycles crossing
| .00005					    | Children crossing

In this the probability was very similiar 0.47 for the right sign. and 0.50 for the wrong one. Because of the higer probability End of no passing was choosen.



