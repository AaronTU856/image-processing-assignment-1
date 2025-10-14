#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Student Name: Aaron Baggot
# Student Number: C22716399
# TU856/4


# In[37]:


from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[38]:


# Save image
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving to:", OUT_DIR.resolve())

def show(title, img):
    plt.figure(figsize=(8,5))
    if img.ndim == 3:
        plt.imshow(cv.cvtColoe(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title);
    plt.axis("off");
    plt.show()


# In[39]:


# Load Image
img_path = Path("/Users/aaronbaggot/Desktop/Image_Processing_Assignment_1/data/raw/Shark 1.PNG")  
# img_path = Path("/Users/aaronbaggot/Desktop/Image_Processing_Assignment_1/data/raw/Shark 2.PNG")

I = cv.imread(str(img_path), cv.IMREAD_COLOR)

if I is None:
    raise Exception(f"Image not found: {img_path}")
h, w, = I.shape[:2]

# Resize x 2
S = cv.resize(I, dsize=(2*w, 2*h), interpolation=cv.INTER_CUBIC)

# Crop top left Tested on different values
#C I[0:min(150, h), 0:min(200, w)]

C = I[0:min(180, h), 0:min(270, w)]

#C = I[0:min(100, h), 0:min(150, w)]


# In[40]:


# Rotate 45 degrees on the same canvas
cx, cy = w // 2, h // 2

M = cv.getRotationMatrix2D((cx, cy), 45, 1.0)

R = cv.warpAffine(I, M, (w, h), flags=cv.INTER_LINEAR, borderMode =cv.BORDER_REFLECT)


# In[41]:


# Display the outputs
plt.show("Original", I)


# In[42]:


plt.show("Upscaled", S)


# In[43]:


plt.show("Top left Cropped", C)


# In[44]:


plt.show("Rotated", R)


# In[ ]:





# In[31]:


# Save each image with discription
cv.imwrite(str(OUT_DIR / "01_Original.png"), I)
cv.imwrite(str(OUT_DIR / "02_Upscaled.png"), S)
cv.imwrite(str(OUT_DIR / "03_Cropped.png"), C)
cv.imwrite(str(OUT_DIR / "04_Rotated.png"), R)
print("Saved Outputs.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




