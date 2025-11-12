# Assignment 1
**Student Name:** Aaron Baggot  
**Student Number:** C22716399  
**Course:** TU856/4  

This assignment builds on the functions we covered in the labs using Python and OpenCV [1]. I tested out a few different approaches, and the one that worked best followed these main steps in the pipeline:

## Bilateral Filter
The bilateral filter helps reduce speckle and patchy noise while keeping the edges sharp. It does this by weighting pixels based on both how close they are in space and how similar their intensity is. Underwater images can be really grainy and uneven, so this step smooths out the water without making the shark's outline blurry [2].  

For example, in the code, it's done with: `den = cv.bilateralFilter(I, 7, 75, 75)`

## CLAHE on the LAB L Channel
Next, I applied CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the L (lightness) channel in the LAB color space. This boosts the local contrast in different parts of the image while preventing noise from getting too amplified by clipping the big peaks. It really helps with low-contrast underwater scenes and brings out details like the shark's body stripes [3].  

In the code, it's implemented as: `L2 = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8)).apply(L)`

## Light Unsharp Mask
After that, I used a small Gaussian blur and a weighted addition to gently sharpen the edges and bring back some definition that might have been lost in the denoising and contrast steps. This keeps the image looking natural. The Gaussian blur gives more weight to pixels closer to the center of the kernel [4].  

Here's how it looks in the code:  
```
blur = cv.GaussianBlur(enh, (0,0), 1.0)
enh = cv.addWeighted(enh, 1.20, blur, -0.20, 0)
```

## Otsu Thresholding for Segmentation
For segmentation, I used Otsu's method, which automatically picks the best threshold for turning the image into black and white. I made two simple versions so the same code can handle different types of images:  

- **Underwater mode:** Otsu on the LAB-b channel (blue-yellow) combined with the inverted HSV saturation to separate the fish from the open water.  
- **Silhouette mode:** Otsu on the inverted grayscale for darker, low-saturation shapes like Shark 2.  

This is done with `cv.threshold` using `THRESH_OTSU` [5].  

Example: `_, mask = cv.threshold(inv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)`

## Morphology and Largest Component
Then, I applied some morphological operations: closing to fill small holes, opening to remove specks, and a light dilation to thicken thin parts of the mask. This gives a clean blob for cropping without needing much tweaking [6]. After that, I labeled the connected components and kept only the largest one to get rid of any stray pixels or bubbles [7].  

In the code:  
```
k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (KSIZE, KSIZE))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=CLOSE_IT)
num, labc = cv.connectedComponents(mask)
```

## Rotate and Crop
For underwater images, I straightened the mask's contour using a minimum-area rectangle and rotated it to be horizontal with `cv.minAreaRect` and `cv.getRotationMatrix2D`. For regular photos, like the cat in snow, I just used a simple axis-aligned bounding box to avoid weird wedge effects. I added a bit of padding so the subject doesn't get cut off [8].  

Example:  
```
rect = cv.minAreaRect(cnt)
M = cv.getRotationMatrix2D((W//2, H//2), rect[-1], 1.0)
I_rot = cv.warpAffine(enh, M, (W, H))
```

## Minimal Background
Finally, I sampled pixels from a thin border outside the mask, took the median color for each channel, and used that as a flat background. This creates the clean, minimal background we needed [9].  

## Summary
Overall, the program takes an image with a plain background and outputs an enhanced version that's clearer and centered on a solid background. The bilateral filter cuts down on noise while keeping edges intact. Otsu thresholding makes a foreground mask using color info for blue underwater shots or inverted grayscale for silhouettes. Small morphological ops clean up the mask, and we keep the biggest blob. Then, the subject gets rotated to horizontal for underwater pics or just cropped for normal ones, and placed on a background color guessed from the image's border. The whole pipeline is short and only uses a few adjustable values from the labs.

## References
[1] J. Howse, P. Joshi, and M. Beyeler, *OpenCV: computer vision projects with Python*, 1st ed. Packt Publishing, 2016.  

[2] C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color images,” in *Sixth International Conference on Computer Vision*, Jan. 1998, pp. 839–846. doi: 10.1109/ICCV.1998.710815.  

[3] J. Wu, “Enhancing Object Sorting Under Low-Light Conditions with CLAHE, Gaussian Blur, ROI, and Custom PID on a Raspberry Pi Robotic Arm,” *Appl. Comput. Eng.*, vol. 96, no. 1, pp. 93–98, Nov. 2024, doi: 10.54254/2755-2721/96/20241240.  

[4] S. Singh, R. Verma, and A. K. Singh, “Image filtration in Python using openCV,” *Turk. J. Comput. Math. Educ.*, vol. 12, no. 6, pp. 5136–5143, 2021.  

[5] M. Huang, W. Yu, and D. Zhu, “An Improved Image Segmentation Algorithm Based on the Otsu Method,” in *2012 13th ACIS International Conference on Software Engineering, Artificial Intelligence, Networking and Parallel/Distributed Computing*, Aug. 2012, pp. 135–139. doi: 10.1109/SNPD.2012.26.  

[6] J. Jenrette, Z. Y.-C. Liu, P. Chimote, T. Hastie, E. Fox, and F. Ferretti, “Shark detection and classification with machine learning,” *Ecol. Inform.*, vol. 69, p. 101673, July 2022, doi: 10.1016/j.ecoinf.2022.101673.  

[7] F. Bolelli, S. Allegretti, and C. Grana, “Quest for Speed: The Epic Saga of Record-Breaking on OpenCV Connected Components Extraction”.  

[8] A. Mordvintsev, “OpenCV-Python Tutorials Documentation”.  

[9] A. P. Athane and D. S. R. Prasad, “Image Enhancement Based on Opencv Using Python 2.7 – Review,” vol. 8, no. 5, 2021.
