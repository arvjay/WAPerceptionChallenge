#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import cv2
from vision import Vision
from PIL import Image, ImageDraw

# Unused imports
#from sklearn.linear_model import HuberRegressor
#from sklearn.preprocessing import StandardScaler
#import numpy as np
#from matplotlib import pyplot as plt


# In[2]:


# Reads original image
path = "original.png";
img = cv2.imread(path);
orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[3]:


# Loads trained classifier
cascade_cone = cv2.CascadeClassifier("cascade\cascade.xml");
# Vision object allows visual representation of object detection (Vision class by @learncodebygaming on github)
vision_cone = Vision();


# In[4]:


# Creates cone detection rectangles
rectangles = cascade_cone.detectMultiScale(orig_img);


# In[5]:


# Adds the centers of the rectangles to respective x and y position lists
x_list = [];
y_list = [];
for rectangle in rectangles:
    x_list.append(rectangle[0]+rectangle[2]/2);
    y_list.append(rectangle[1]+rectangle[3]/2);
# Separates the cone positions on the left side from the positions on the right side by comparing x values
x_list_left = [];
x_list_right = [];
y_list_left = [];
y_list_right = [];
for i,x in enumerate(x_list):
    if x > orig_img.shape[1]/2:
        x_list_right.append(x);
    else:
        x_list_left.append(x);
        
# Appends the y value to the corresponding list based on whether the x position is on the left or right
for i,position in enumerate(rectangles):
    y_pos = position[1]+position[3]/2;
    if position[0]+position[2]/2 in x_list_left:
        y_list_left.append(y_pos);
    else:
        y_list_right.append(y_pos);
        
# Sorts the list positions to make it easier to access the smallest and largest positions needed to draw the lines
x_list_left.sort();
x_list_right.sort();
y_list_left.sort();
y_list_right.sort();

# Tried using Huber Linear Regression model to draw the lines but couldn't find a way to integrate it with the original image
# Credit: https://stackoverflow.com/questions/61143998/numpy-best-fit-line-with-outliers
 
# standardize
#x_left_scaler, y_left_scaler = StandardScaler(), StandardScaler();
#x_left_train = x_left_scaler.fit_transform(x_list_left_np[..., None]);
#y_left_train = y_left_scaler.fit_transform(y_list_left_np[..., None]);

# fit model
#left_model = HuberRegressor(epsilon=1);
#left_model.fit(x_left_train, y_left_train.ravel());

# do some predictions
#test_x_left = np.array([100, 850])
#print(test_x_left)
#print(left_model.predict(x_left_scaler.transform(test_x_left[..., None])))
#predictions_left = y_left_scaler.inverse_transform((
#    left_model.predict(x_left_scaler.transform(test_x_left[..., None])).reshape(-1,1))
#)

# standardize
#x_right_scaler, y_right_scaler = StandardScaler(), StandardScaler();
#x_right_train = x_right_scaler.fit_transform(x_list_right_np[..., None]);
#y_right_train = y_right_scaler.fit_transform(y_list_right_np[..., None]);

# fit model
#right_model = HuberRegressor(epsilon=1);
#right_model.fit(x_right_train, y_right_train.ravel());

# do some predictions
#test_x_right = np.array([950, 1700])
#predictions_right = y_right_scaler.inverse_transform((
#    right_model.predict(x_right_scaler.transform(test_x_right[..., None])).reshape(-1,1))
#)


# In[6]:


# Plots the lines on the image using Pillow library
image = Image.open("original.png");
draw = ImageDraw.Draw(image);
draw.line([x_list_left[1], y_list_left[-1], x_list_left[-1], y_list_left[0]], width = 4, fill = 'red');
draw.line([x_list_right[-1], y_list_right[-1], x_list_right[0], y_list_right[0]], width = 4, fill = 'red');
image.save("answer.png");


# In[7]:


# Draws rectangles to visually represent object detection and saves it to png file for reference (draw_rectangles() method sourced from @learncodebygaming on github)
detection_image = vision_cone.draw_rectangles(orig_img, rectangles);
detection_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("detection_image.png",detection_image);

