#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Student Name: Aaron Baggot
# Student Number: C22716399
# TU856/4


# In[17]:


from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


# Save image
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving to:", OUT_DIR.resolve())

def show(title, img):
    plt.figure(figsize=(8,5))
    if img.ndim == 3:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title);
    plt.axis("off");
    plt.show()


# In[19]:


# Load Image
img_path = Path("/Users/aaronbaggot/Desktop/Image_Processing_Assignment_1/data/raw/Shark 1.PNG")  
# img_path = Path("data/raw/Shark 2.PNG")

I = cv.imread(str(img_path), cv.IMREAD_COLOR)

if I is None:
    raise Exception(f"Image not found: {img_path}")
h, w, = I.shape[:2]

# Resize x 2
S = cv.resize(I, dsize=(2*w, 2*h), interpolation=cv.INTER_CUBIC)

# Crop top left Tested on different values
# C = I[0:min(150, h), 0:min(200, w)]

# C = I[0:min(180, h), 0:min(270, w)]

C = I[0:min(100, h), 0:min(150, w)]


# In[20]:


# Rotate 45 degrees on the same canvas
cx, cy = w // 2, h // 2

M = cv.getRotationMatrix2D((cx, cy), 45, 1.0)

R = cv.warpAffine(I, M, (w, h), flags=cv.INTER_LINEAR, borderMode =cv.BORDER_REFLECT)


# In[21]:


# Display the outputs
show("Original", I)


# In[22]:


show("Upscaled", S)


# In[23]:


show("Top left Cropped", C)


# In[24]:


show("Rotated", R)


# In[25]:


print("Saved outputs")


# In[26]:


# Save each image with discription
cv.imwrite(str(OUT_DIR / "01_Original.png"), I)
cv.imwrite(str(OUT_DIR / "02_Upscaled.png"), S)
cv.imwrite(str(OUT_DIR / "03_Cropped.png"), C)
cv.imwrite(str(OUT_DIR / "04_Rotated.png"), R)
print("Saved Outputs.")


# In[28]:


# Denoise whcih keeps edges, contrast via CLAHE
den = cv.bilateralFilter(I, d=7, sigmaColor=75, sigmaSpace=75)

LAB = cv.cvtColor(den, cv.COLOR_BGR2LAB)
L, A, B = cv.split(LAB)
L2 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(L)
enh = cv.cvtColor(cv.merge([L2, A, B]), cv.COLOR_LAB2BGR)

blur = cv.GaussianBlur(enh, (0,0), 1.0)
enh = cv.addWeighted(enh, 1.4, blur, -0.4, 0)

plt.show("Enhanced", enh)
cv.imwrite(str(OUT_DIR / "05_enhanced.png"), enh)




# In[33]:


# Segment, Otsu on S & LAB-b, morphology, largest component
hsv = cv.cvtColor(enh, cv.COLOR_BGR2HSV); H,S,V = cv.split(hsv)
lab = cv.cvtColor(enh, cv.COLOR_BGR2LAB); L, a, b = cv.split(lab)

_, th_s = cv.threshold(S, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
_, th_b = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

mask = cv.bitwise_and(th_s, th_b)

k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, iterations=1)

num, labels = cv.connectedComponents(mask)
if num > 1:
    sizes = [(labels==i).sum() for i in range(1, num)]
    mask = np.where(labels==(1+int(np.argmax(sizes))), 255, 0).astype(np.uint8)

show("Mask, after mophology and largest blob)", mask)
cv.imwrite(str(OUT_DIR / "06_mask.png"), mask)


# In[38]:


# Rotate horizontal and crop tight
cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
if cnts:
    cnt = max(cnts, key=cv.contourArea)
    rect = cv.minAreaRect(cnt)
    angle = rect[-1]
    if angle < -45: angle += 90
    M = cv.getRotationMatrix2D((w//2, h//2),angle, 1.0)
    I_rot = cv.warpAffine(enh, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    m_rot = cv.warpAffine(mask, M, (w, h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT)
    pts = cv.findNonZero(m_rot)
    x,y,ww,hh = cv.boundingRect(pts) if pts is not None else (0,0,w,h)
    crop = I_rot[y:y+hh, x:x+ww]
    mcrp = m_rot[y:y+hh, x:x+ww]
else:
    crop, mcrp = enh, mask


show("Rotated image", I_rot)
show("Cropped to fish", crop)
cv.imwrite(str(OUT_DIR / "07_rotated.png"), I_rot)
cv.imwrite(str(OUT_DIR / "08_cropped.png"), crop)


# Compose clean background with solid median for borders
inv = (mcrp==0).astype(np.uint8)*255
border = np.zeros_like(inv)
border[10,:]=255; 
border[-10:,:]=255;
border[10,:]=255; 
border[:,-10:]=255


samples = cv.bitwise_and(crop, crop, mask=cv.bitwise_and(inv, border)).reshape(-1, 3)
samples = samples[(samples.sum(1) > 0)]
color = tuple(map(int, (np.median(samples,0) if len(samples) else np.array([210,220,230]))))

canvas = np.full_like(crop, color, dtype=np.uint8)
final = cv.add(cv.bitwise_and(crop, crop, mask=mcrp),
               cv.bitwise_and(canvas, canvas, mask=cv.bitwise_not(mcrp)))

plt.show("Final Clean background", final)
cv.imwrite(str(OUT_DIR /"07_final.png"), final)
print("All saved in:", OUT_DIR)







# In[ ]:





# In[ ]:




