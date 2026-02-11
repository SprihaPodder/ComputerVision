#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Error: Cannot open webcam!" << endl;
        return -1;
    }

    Mat frame, background;

    cout << "Stand aside. Capturing background in 3 seconds..." << endl;

    // Capture background
    for (int i = 0; i < 60; i++) {
        cap >> frame;
        imshow("Capturing Background", frame);
        waitKey(30);
    }

    background = frame.clone();
    cout << "Background captured!" << endl;

    while (true) {

        cap >> frame;
        if (frame.empty()) break;

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Detect black color (tune if needed)
        Scalar lowerBlack(0, 0, 0);
        Scalar upperBlack(180, 255, 60);

        Mat mask;
        inRange(hsv, lowerBlack, upperBlack, mask);

        // Clean mask
        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

        morphologyEx(mask, mask, MORPH_DILATE,
                     getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

        // Invert mask
        Mat inverseMask;
        bitwise_not(mask, inverseMask);

        // Extract visible person
        Mat visiblePart;
        bitwise_and(frame, frame, visiblePart, inverseMask);

        // Extract background where black cloth is
        Mat backgroundPart;
        bitwise_and(background, background,
                    backgroundPart, mask);

        // Combine both
        Mat result = visiblePart + backgroundPart;

        imshow("Invisibility Cloak Effect", result);

        char key = waitKey(30);
        if (key == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}