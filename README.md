## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/test2.jpg "Original"
[image2]: ./camera_cal/calibration2.jpg "Original Calibration Image"
[image3]: ./output_images/calibration2undistort.jpg "Undistorted Calibration Image"
[image4]: ./output_images/test2undistort.jpg "Undistorted Image"
[image5]: ./output_images/test2binarythres1.jpg "Binary Threshold"
[image6]: ./output_images/test2binarythres2.jpg "Binary and Color Threshold"
[image7]: ./output_images/test2binarythres3ROI.jpg "Binary and Color Threshold with ROI"
[image8]: ./output_images/straight_lines1perspectpts.jpg "Perspective Transform Points"
[image9]: ./output_images/straight_lines1warped.jpg "Straight Lines Warped"
[image10]: ./output_images/straight_lines1warpedbinary.jpg "Binary Straight Lines Warped"
[image11]: ./output_images/lanelines.jpg "Lane Lines Detection"
[image12]: ./output_images/test2final.jpg "Overlayed Image"
[video1]: ./test_videos_output/project_video.mp4 "Video"

<!--- ## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. The code for the project is located in the jupyter notebook "Advanced Lane Lines.ipynb." ---> 

---
### Project Code
The project code is given in [Advanced Lane Lines.ipynb](https://github.com/anammy/CarND-Advanced-Lane-Lines/blob/master/Advanced%20Lane%20Lines.ipynb)

### Camera Calibration

#### 1. Camera matrix and distortion coefficients

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The functions 'cv2.findChessboardCorners()' and 'cv2.drawChessboardCorners()' was used to detect the pixel positions for the chessboard corners and draw them back onto the image.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the calibration images using the `cv2.undistort()` function and obtained this result: 

*Original Calibration Image*     |  *Undistorted Calibration Image*
:-------------------------:|:-------------------------:
![alt text][image2] | ![alt text][image3]

### Pipeline (single images)

#### 1. Distortion-corrected image

Using the camera matrix and distortion coefficients from the calibration step, the following test image was undistorted.

*Original Test Image*
![alt text][image1]

*Undistorted Test Image*
![alt text][image4]

#### 2. Binary Threshold Image

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of the binary gradient threshold image.

*Binary Gradient Threshold*
![alt text][image5]

Yellow lane lines were not extracted in all the test images using only binary gradient threshold. As a result, the test images were converted to HLS color space and color thresholding was applied to the S-channel. The yellow lane lines in the test images were more robustly detected using a combined binary gradient and color thresholding methodology.

*Binary Gradient and Color Threshold*
![alt text][image6]

This image was then processed to extract out the region of interest containing the lane lines. 

*Region of Interest Binary Threshold Image*
![alt text][image7]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `unwarp_transform`. The `unwarp_transform()` function takes as inputs an image (`img`), as well as camera calilbration matrices.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[205, img_size[1] - 1],
    [596, 450],
    [685, 450],
    [1105, img_size[1] - 1]])

offset = 400

dst = np.float32(
    [[offset, img_size[1]],
    [offset, 0],
    [img_size[0] - offset, 0],
    [(img_size[0] - offset, img_size[1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 719      | 400, 720        | 
| 596, 450      | 400, 0      |
| 685, 450     | 880, 0      |
| 1105, 719      | 880, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

*Perspective Transformation Source Points*
![alt text][image8]

*Perspective Transformation (Color)*
![alt text][image9]

*Perspective Transformation (Binary)*
![alt text][image10]

#### 4. Lane Line Identification and Fitting

Using the function 'fit_lines()', I used the code from the classes to implement a sliding window search for the lane lines and fit the points with a 2nd order polynomial.

*Lane Line Identification*
![alt text][image11]

#### 5. Lane Line Properties and Vehicle Position

I calculated the radius of curvature of the lane lines and the position of the vehicle with respect to the center in the function 'Radius_CenterDist().'

#### 6. Overlayed Image

I then overlayed the lane lines onto the original test image along with the lane radius of curvature and vehicle position using the function 'VisualonImage'.  Here is an example of my result on a test image:

*Overlayed Image*
![alt text][image12]

---

### Pipeline (video)

#### 1. Project Video

I applied the above techniques to the frames in the video 'project_video.mp4' using the function 'process_image().'

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

I used the following strategies to make the video analysis pipeline more robust. Once a frame is fitted with lane lines, the lane lines in the next frames are searched in the vicinity of the previous fitted points using the function 'fit_nextline()' using the code from the classes. Some sanity checks performed on the fitted lines include checking that the radius of curvature of the two lines are similar and the distance between the left and right lane lines detected were approximately equal to the standard lane width in the US of 3.7m.

If the detected lane lines fail the sanity check, the lane lines from the previous frame are used. If 2 consecutive frames fail the sanity checks, the original line search using the function 'fit_lines()' is initiated. In addition, the lane lines projected onto the video frames are also averaged over the last five frames in order to smooth out and reduce the shakiness of the lane lines drawn.

The pipeline currently does have difficulty in maintaining stable measurements when the pavement material in the project video changes due to spurious line detections. In the future, I will implement a check to see if the right and left lanes are roughly parallel along its projected length and compare radii of curvature between frames to check for unrealistic rapid changes.
