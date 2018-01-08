## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted]: ./writeup/undistorted.png "Undistorted"
[undistorted2]: ./writeup/undistorted2.png "Undistorted"
[threshold]: ./writeup/threshold.png "Thresholding"
[lineident]: ./writeup/line_ident.png "Line identification"

[transform]: ./writeup/transform.png "Perspective transform"
[backprojected]: ./writeup/backprojected.png "Projected back"

[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "CameraCalibration.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to some test images using the `cv2.undistort()` function (next coding cell) and obtained results like the following: 

![alt text][undistorted]

### Pipeline (single images)

I created a new IPython Notebook for the rest of the project. "AdvancedLaneFinding.ipynb"

#### 1. Distortion correction example
In this section, i used the camera calibration and distortion correction from prior exercise to correct input images. As an example, it was applied to one of the test images. The result can be seen in the following image.

![alt text][undistorted2]

#### 2. Thresholding

I used a combination of color and gradient thresholds to generate a binary image (function `threshold` in the third coding cell of the notebook).  Here's an example of my output for this step. 

![alt text][threshold]

The following steps were applied: 

* The input image is converted to its HSL representation
* Gradients in x-direction are being computed using a sobel operation
* The generated binary images keeps pixels having a saturation value between 90 and 200, *or* a sobel gradient between 50 and 100
* From the resulting pddixels, only those are being kept having a lightness value of at least 35 *or* an OpenCV hue value between 0 and 60

#### 3. Perspective transformation

The code for my perspective transform includes a function called `transform()`, which appears in the 4th code cell of the IPython notebook.  The function takes as inputs an image (`img`). It computes the transformation using a perspective transformation matrix that was generated with hard-coded values as follows:

```python
src_corners = np.array([[585, 460], [698, 460], [995, 659], [296, 659]], np.float32) #2 686
dst_corners = np.array([[300, 0], [1280 - 300, 0], [1280 - 300, 720], [300, 720]], np.float32)
M = cv2.getPerspectiveTransform(src_corners, dst_corners)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 300, 0        | 
| 698, 460      | 980, 0         |
| 995, 659      | 980, 720      |
| 296, 659      | 300, 720        |

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][transform]

#### 4. Identification of lane line pixels

In the next part of my notebook I identify lane line pixels. This is done in the function `find_lines`. I use a sliding window search to identify lane line pixels. In the left and right lane.

In a first step, I check a histogram of the lower half of an input image to identify at which x position most pixels available after thresholding appear. 

The first two windows start in the lower quarter of the input image. Each window has a height of 90 and a width of 180. At each y position of a window, the window is repositioned based on the pixels having the value 1 being contained. If a minimum number of these pixels is contained, the window is being recentered, taking their average as the average window position.  New windows are being calculated moving both down and up from the first window's position. If a window on the left lane is not being recentered, its x position is the same as for the last window. For the right lane, if a window is not being recentered, its center is based on the window on the left lane. The distance of this window to the corresponding window on the left lane is the same as the distance of the two initial windows.


After all windows have been computed, I fit a second order polynomial to the center pixels of the left and right windows respectively.

![alt text][lineident]

#### 5. Curvature radius; Offset

These values are being calculated in the function mentioned previously as follows:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
left_fit_cr  = np.polyfit(np.array(left_ys) *ym_per_pix, np.array(left_xs) *xm_per_pix, 2)
right_fit_cr = np.polyfit(np.array(right_ys)*ym_per_pix, np.array(right_xs)*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)  / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
rad = (left_curverad + right_curverad) /2 # radius
my_center_x = 1280 / 2
lane_center_x = right_xs[1] - left_xs[1]  
center_distance = my_center_x - lane_center_x
offset = xm_per_pix * center_distance
```
For calculating the center offset, it's being assumed that the car's camera is positioned at the center of the car.


#### 6. Projecting back the resulting image 

I implemented this step in the function `create_annotated_image`. The inverse of the transformation matrix mentioned earlier is being applied.

![alt text][backprojected]


---

### Pipeline (video)


Here's a [link to my video result](./output_project_video.mp4).

---

### Discussion


