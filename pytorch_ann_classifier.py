# -*- coding: utf-8 -*-
"""

## **Mounting Google Drive**
First, we need to import the dataset of images from your Google Drive. To do so, **run the below cell**. This will mount your Drive to the running Colab instance. Then, you will be able to access all your Google Drive data in this notebook.
"""

from google.colab import drive
drive.mount('/content/drive')

"""## **Import the Libraries**
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

"""## **Importing the Dataset**

We now need to import the dataset from the Google Drive. Below given is a function named `getdata`, which takes the path of the folder from which data is to be imported as its argument.
You are required to give in the path of the folder in which the images of `pizza` and `not_pizza` are saved in your Google Drive. 
"""

def getdata(path):
  data = torch.tensor([])
  file_list = os.listdir(path)
  for file_name in file_list[:350]:
    image_path = os.path.join(path, file_name)
    image = mpimg.imread(image_path)
    imageData = torch.from_numpy(image).long()
    data = torch.cat((data, imageData.unsqueeze(0)), dim=0)
  return data


pizza_path = 'drive/MyDrive/pizza_vs_not/pizza'
not_pizza_path = 'drive/MyDrive/pizza_vs_not/not_pizza'


not_pizza_data = getdata(not_pizza_path)
pizza_data = getdata(pizza_path)

"""Using the next cell, you can preview the images you've just loaded"""


index = 222 # 0 to 349
data = not_pizza_data # pizza_data (or) not_pizza_data


plt.imshow(data[index].int())

"""## **Preprocessing the dataset**

Before we begin working with the neural network, we need to make sure our data (in this case, $64\times64$ RGB images) is in a format that neural networks can work with.

First, we create a `train_data` collection of images for training the neural network, and another collection `test_data` that will then be used to check the accuracy of the trained neural network. We do this by taking slices of the total data and concatenating them.  
"""

train_data = torch.cat((pizza_data[:300], not_pizza_data[:200]), dim = 0)
test_data = torch.cat((pizza_data[300:350], not_pizza_data[200:250]), dim = 0)
print(train_data.shape)
#dim = 0 is used so that train and test data are stored in 2 separate tensors
#consider it as if dim = 0 rows space is unalterd and if dim = 1 columns space is unaltered

"""Each image is still stored as a $64\times64\times3$ tensor ie. a $64\times64$ array of 3 numbers - the RGB values of the pixel, taking values between **0 to 255**.  
You need to flatten the images in both datasets to make *reshaped* data, and normalise them to get the *final* data we will use to train and test the model.

<details>
  <summary>Hint</summary>
  Use the reshape command to flatten the dataset and then normalise the flattened dataset (ie. make sure all the values lie between 0 and 1).
</details>
"""

reshaped_train_data = train_data.reshape([500,64*64*3])
reshaped_test_data = test_data.reshape([100,64*64*3])

final_train_data = reshaped_train_data/255
final_test_data = reshaped_test_data/255


print(final_train_data.shape, final_test_data.shape)

"""Now that you have made two datasets, you need to make their corresponding `labels` Tensors, which store the true output for each image (whether or not it is a pizza).

In the `labels` Tensor, use **`1`** for images that belong to `pizza` and **`0`** for `not_pizza`. (Try to use the concatenate function instead of simply using *for* loops for generating the `labels` Tensor !!)

**Remember** to ensure that both `train_labels` and `test_labels` are 2D tensors of appropriate dimensions. Otherwise, it can cause issues ahead.
"""


train_labels = torch.cat((torch.ones(300,1),torch.zeros(200,1)),0)
test_labels = torch.cat((torch.ones(50,1),torch.zeros(50,1)),0)


print(train_labels.shape, test_labels.shape)

"""##**Building the Neural Network**

First, initialise the hyperparameters of the neural network.  
Here, we are going to make a 3 layer neural network (i.e., 1 input layer, 2 hidden layers and 1 output layer). The first hidden layer will have 10 nodes, while the second will have 12.   
Enter the number of input parameters, number of nodes of each hidden layer and the number of output parameters.
"""


D_in = 64*64*3
H1 = 10
H2 = 12
D_out = 1
#since it is pizza or not pizza hence we can use only one neuron for this purpose as it ensures both are independent

"""Now you will make the actual model. The Model includes the use of `Linear` function at each layer, alongside non-linear activation functions.

We are going to use `ReLU` functions as activation functions for the input and the first hidden layer and `Sigmoid` function for the final output, as we want an output between 0 and 1.

(Hint: `Linear`, `ReLU` and `Sigmoid` functions are a part of the `nn` module of the **PyTorch** library)
"""

model = torch.nn.Sequential(
    # Do not hard-code any values, use the variables from the previous cell

    nn.Linear(D_in,H1),
    nn.ReLU(),
    nn.Linear(H1,H2),
    nn.ReLU(),
    nn.Linear(H2,D_out),
    nn.Sigmoid()
)

model(final_train_data[0]) #To check if the model works
#computation graph and has a gradient function (grad_fn) assigned to it, which tracks operations performed on tensors to facilitate automatic differentiation.

"""After making the model, we defined the `loss_fn` , i.e., the loss function as the Binary Cross Entropy Loss.

Here, you will implement gradient descent.

The `learning_rate` is the step size which the neural network takes when it updates the parameters of the network. (Try to explore various values of step size. Values similar to 0.005 are usually suitable). Also play around with the number of `iterations` the gradient descent algorithm needs to take.

**Follow the steps given as per the comments.**

*Note: When you run the code, it could take about 5 minutes for the network to finish training.*
"""

loss_fn = nn.BCELoss()

learning_rate = 0.005
iterations = 100

for t in range(iterations):
  for i in range(500):
    # call the model on the dataset
    y_pred = model(final_train_data[i])

    #calculate the loss
    loss =  loss_fn(y_pred,train_labels[i])


    if (t%1000)%100 == 0:
        print(loss)

    # calculate the gradients
    loss.backward()

    # update the values of the parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # reset the gradients
    model.zero_grad()

torch.save(model, 'model_best.pt')

y_pred = model(final_train_data[0])
y_pred.shape == train_labels[0].shape

"""## **Checking the Results**
Now that we have used the training set to train the network, we shall use the test set to check how the neural network performs with new inputs.
"""

index = 4 #0 to 99

plt.imshow(test_data[index].int())
print (f'According to the neural network, index = {index} is {"a pizza" if model(final_test_data[index]) > 0.5 else "not a pizza"}' )

"""To quantize how accurately the neural network is able to classify images, we have defined a helper `predict` function that takes a dataset and returns the fraction of times the neural network correctly classified the image.  

Complete the function such that it prints the correct accuracy of its predictions.
"""

def predict(model, data, labels):

    probabilities = model(data)

    # generate the predictions tensor using the probabilities variable, which indicates the prediction made by the model for the given data using 0 and 1

    predictions = torch.where(probabilities > 0.5, torch.ones_like(probabilities), torch.zeros_like(probabilities))

    print("Accuracy: "  + str(torch.sum((predictions == labels)).item()/predictions.shape[0]))

"""**Run** the cell below to find out the accuracy of your model for the training and test datasets."""

predict(model, final_train_data, train_labels)
predict(model,final_test_data,test_labels)
print("Done!")
"""
