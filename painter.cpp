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
  
  // Create a canvas to draw the path
  Mat canvas;
  
  // Store previous centroid position
  Point prevCentroid = Point(-1, -1);
  
  // Define color rectangles on the right side
  int frameWidth = 0, frameHeight = 0;
  Rect redRect, greenRect, blueRect, clearRect;
  Scalar activeColor;
  
  while(1){
 
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    
    // Initialize rectangles on first frame
    if(frameWidth == 0){
      frameWidth = frame.cols;
      frameHeight = frame.rows;
      
      // Create clear rectangle on top left
      clearRect = Rect(10, 10, 80, 60);
      
      // Create rectangles on the right side of the frame
      int rectWidth = 80;
      int rectHeight = frameHeight / 3;
      int startX = frameWidth - rectWidth - 10;
      
      redRect = Rect(startX, 0, rectWidth, rectHeight);
      greenRect = Rect(startX, rectHeight, rectWidth, rectHeight);
      blueRect = Rect(startX, 2 * rectHeight, rectWidth, rectHeight);
      
      // Initialize canvas
      canvas = Mat::zeros(frame.size(), frame.type());
    }
 
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
    
    // Calculate centroid of red pixels using moments
    Moments M = moments(redMask, true);
    
    // Calculate centroid coordinates
    double cx = M.m10 / (M.m00 + 1e-5);  // x-coordinate of centroid
    double cy = M.m01 / (M.m00 + 1e-5);  // y-coordinate of centroid
    
    // Create a copy of frame to draw on
    Mat frameWithCentroid = frame.clone();
    
    // Draw the three color rectangles
    rectangle(frameWithCentroid, redRect, Scalar(0, 0, 255), -1);      // Red rectangle
    rectangle(frameWithCentroid, greenRect, Scalar(0, 255, 0), -1);    // Green rectangle
    rectangle(frameWithCentroid, blueRect, Scalar(255, 0, 0), -1);     // Blue rectangle
    
    // Draw clear button
    rectangle(frameWithCentroid, clearRect, Scalar(200, 200, 200), -1); // Gray rectangle
    putText(frameWithCentroid, "CLEAR", Point(clearRect.x + 10, clearRect.y + 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    
    // Overlay canvas paths with proper alpha blending
    addWeighted(frameWithCentroid, 1.0, canvas, 0.7, 0, frameWithCentroid);
    
    // Draw green circle at centroid if red pixels are detected
    if(M.m00 > 0){
      Point centroidPt = Point(int(cx), int(cy));
      
      // Check if centroid is in clear button
      if(clearRect.contains(centroidPt)){
        canvas = Mat::zeros(frame.size(), frame.type());
        prevCentroid = Point(-1, -1);
        activeColor = Scalar(255, 255, 255);  // Reset to white
        cout << "Canvas cleared!" << endl;
      }
      // Check if centroid is in any of the color rectangles to change active color
      else if(redRect.contains(centroidPt)){
        activeColor = Scalar(0, 0, 255);  // Red
        cout << "Active color: RED" << endl;
      }
      else if(greenRect.contains(centroidPt)){
        activeColor = Scalar(0, 255, 0);  // Green
        cout << "Active color: GREEN" << endl;
      }
      else if(blueRect.contains(centroidPt)){
        activeColor = Scalar(255, 0, 0);  // Blue
        cout << "Active color: BLUE" << endl;
      }
      
      // Draw line from previous centroid to current centroid with active color (everywhere on screen)
      if(prevCentroid.x != -1 && prevCentroid.y != -1){
        line(canvas, prevCentroid, centroidPt, activeColor, 4);
      }
      
      // Update previous centroid
      prevCentroid = centroidPt;
      
      // Draw green circle at centroid
      circle(frameWithCentroid, centroidPt, 10, Scalar(0, 255, 0), 2);  // Green circle
      circle(frameWithCentroid, centroidPt, 2, Scalar(0, 255, 0), -1);  // Filled center point
      
      // Print centroid coordinates on console
      cout << "Centroid: (" << cx << ", " << cy << ")" << endl;
    }
    
    // Display frame with centroid marker and path
    imshow("Remote Painter", frameWithCentroid);
 
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
    // Press 'c' to clear the canvas
    if(c=='c' || c=='C'){
      canvas = Mat::zeros(frame.size(), frame.type());
      prevCentroid = Point(-1, -1);
    }
  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
   
  return 0;
}