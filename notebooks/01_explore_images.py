#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Student Name: Aaron Baggot
# Student Number: C22716399
# TU856/4


# In[10]:


import glob
print(glob.glob("data/raw/*"))
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


# In[11]:


# Open image
img_path = Path("/Users/aaronbaggot/Desktop/Image_Processing_Assignment_1/data/raw/Shark 1.PNG")

# Open image in greyscale
I = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if I is None:
    raise exception(f"Image not found: {image_path}")




# In[12]:


plt.figure(figsize=(12,5))

plt.imshow(I, cmap="gray")
plt.title("Original Image")


# In[13]:


# Open image
img_path = Path("/Users/aaronbaggot/Desktop/Image_Processing_Assignment_1/data/raw/Shark 1.PNG")

# Open image in greyscale
I = cv2.imread(str(img_path))


# In[14]:


# Get the shape of the image

h, w, d = I.shape

# Resize and make twice as large
S = cv2.resize(I, dsize = (2 * w, 2 * h))

# Crop a small section of the image
C = I[0:180, 0:270]

# Find the centre point of the image
cx, cy = w // 2, h //2

# Rotation angle and scale
angle = 45
scale = 1.0

#create rotation matrix
M = cv2.getRotationMatrix2D(center=(cx,cy), angle = angle, scale = scale)
R = cv2.warpAffine(I, M = M, dsize=(w,h))

# Apply rotation
R = cv2.warpAffine(I, M, dsize=(w, h))


# In[15]:


# Display Images
plt.figure(figsize=(12,5))


plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis("off")


# In[ ]:





# In[16]:


# Edge Detection
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(, (x1, y1), (x2, y2), (255,0,0), 3)

plot.imshow()


# In[ ]:




