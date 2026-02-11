#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

// Generate timestamp string for filenames
string getTimestamp() {
    time_t now = time(0);
    tm *ltm = localtime(&now);

    stringstream ss;
    ss << 1900 + ltm->tm_year
       << setw(2) << setfill('0') << 1 + ltm->tm_mon
       << setw(2) << setfill('0') << ltm->tm_mday << "_"
       << setw(2) << setfill('0') << ltm->tm_hour
       << setw(2) << setfill('0') << ltm->tm_min
       << setw(2) << setfill('0') << ltm->tm_sec;

    return ss.str();
}

int main() {

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Error: Cannot open webcam!" << endl;
        return -1;
    }

    cout << "Press SPACE to capture clean background..." << endl;

    Mat frame, background, grayFrame, grayBackground;

    // Capture initial background
    while (true) {
        cap >> frame;
        imshow("Capture Background (Press SPACE)", frame);

        if (waitKey(30) == 32) { // SPACE
            background = frame.clone();
            break;
        }
    }

    cvtColor(background, grayBackground, COLOR_BGR2GRAY);
    GaussianBlur(grayBackground, grayBackground, Size(5,5), 0);

    cout << "Monitoring started..." << endl;

    const int pixelThreshold = 25;
    const int alarmThreshold = 5000;
    const double minContourArea = 1000.0;

    time_t lastSaveTime = 0;
    const int saveCooldown = 3; // seconds

    while (true) {

        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        GaussianBlur(grayFrame, grayFrame, Size(5,5), 0);

        // Squared difference
        Mat diff;
        absdiff(grayBackground, grayFrame, diff);
        diff.convertTo(diff, CV_32F);
        multiply(diff, diff, diff);
        diff.convertTo(diff, CV_8U);

        // Threshold
        Mat mask;
        threshold(diff, mask, pixelThreshold, 255, THRESH_BINARY);

        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(3,3)));

        int motionPixels = countNonZero(mask);
        bool alarm = motionPixels > alarmThreshold;

        // ===== Contour Detection =====
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool objectDetected = false;

        for (auto &c : contours) {
            if (contourArea(c) > minContourArea) {
                Rect box = boundingRect(c);
                rectangle(frame, box, Scalar(0,255,0), 2);
                objectDetected = true;
            }
        }

        // ===== Alarm + Save Image =====
        if (alarm && objectDetected) {

            putText(frame, "ALARM! INTRUSION DETECTED",
                    Point(40, 50),
                    FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar(0, 0, 255),
                    2);

            time_t now = time(0);

            if (difftime(now, lastSaveTime) >= saveCooldown) {

                string filename = "intrusion_" + getTimestamp() + ".jpg";
                imwrite(filename, frame);

                cout << "Saved: " << filename << endl;
                lastSaveTime = now;
            }
        }

        // Display
        imshow("Live Feed", frame);
        imshow("Detection Mask", mask);

        char key = waitKey(30);

        if (key == 'q') break;

        if (key == 'r') {
            cout << "Resetting background..." << endl;
            background = frame.clone();
            cvtColor(background, grayBackground, COLOR_BGR2GRAY);
            GaussianBlur(grayBackground, grayBackground, Size(5,5), 0);
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}