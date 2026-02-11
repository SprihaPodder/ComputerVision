#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;
 
int main(){
 
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap(0);
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
   
  while(1){
 
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    // Convert BGR to HSV color space (HSV is better for color detection)
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    // Define range for red color in HSV
    // Red color wraps around in HSV, so we need two ranges
    Scalar lowerRed1 = Scalar(0, 100, 100);      // Lower bound for red (0-10)
    Scalar upperRed1 = Scalar(10, 255, 255);     // Upper bound for red (0-10)
    
    Scalar lowerRed2 = Scalar(170, 100, 100);    // Lower bound for red (170-180)
    Scalar upperRed2 = Scalar(180, 255, 255);    // Upper bound for red (170-180)
    
    // Create masks for red color
    Mat mask1, mask2;
    inRange(hsv, lowerRed1, upperRed1, mask1);
    inRange(hsv, lowerRed2, upperRed2, mask2);
    
    // Combine both masks
    Mat redMask = mask1 | mask2;
    
    // Apply morphological operations to reduce noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
    morphologyEx(redMask, redMask, MORPH_OPEN, kernel);
    
    // Create output image showing only red pixels
    Mat redPixels = Mat::zeros(frame.size(), frame.type());
    frame.copyTo(redPixels, redMask);
    
    // Calculate centroid of red pixels using moments
    Moments M = moments(redMask, true);
    
    // Calculate centroid coordinates
    double cx = M.m10 / (M.m00 + 1e-5);  // x-coordinate of centroid
    double cy = M.m01 / (M.m00 + 1e-5);  // y-coordinate of centroid
    
    // Create a copy of frame to draw on
    Mat frameWithCentroid = frame.clone();
    
    // Draw green circle at centroid if red pixels are detected
    if(M.m00 > 0){
      Point centroidPt = Point(int(cx), int(cy));
      circle(frameWithCentroid, centroidPt, 10, Scalar(0, 255, 0), 2);  // Green circle, radius 10, thickness 2
      circle(frameWithCentroid, centroidPt, 2, Scalar(0, 255, 0), -1);  // Filled center point
      
      // Print centroid coordinates on console
      cout << "Centroid: (" << cx << ", " << cy << ")" << endl;
    }
    
    // Display original frame
    imshow("Original Frame", frame);
    
    // Display red color mask
    imshow("Red Mask", redMask);
    
    // Display only red pixels from the original frame
    imshow("Red Pixels Only", redPixels);
    
    // Display frame with centroid marker
    imshow("Red Centroid", frameWithCentroid);
 
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
  
  return 0;
}