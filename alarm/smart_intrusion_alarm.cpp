#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

using namespace cv;
using namespace std;

// 🔊 Play WAV alarm
void playAlarmSound() {
#ifdef _WIN32
    PlaySound(TEXT("alarm.wav"), NULL, SND_FILENAME | SND_ASYNC);
#else
    system("aplay alarm.wav &");
#endif
}

// Timestamp generator
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
        cout << "Cannot open webcam!" << endl;
        return -1;
    }

    cout << "Stand aside. Capturing background..." << endl;

    Mat frame, gray, backgroundFloat, background;

    // Capture initial background
    for (int i = 0; i < 60; i++) {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        gray.convertTo(backgroundFloat, CV_32F);
        imshow("Capturing Background", frame);
        waitKey(30);
    }

    cout << "Monitoring started..." << endl;

    const int pixelThreshold = 45;      // ignore tiny pixel noise
    const int alarmThreshold = 20000;   // require large motion
    const double minContourArea = 4000; // only big objects

    const double learningRate = 0.005;  // slow background update

    time_t lastSaveTime = 0;
    time_t lastSoundTime = 0;

    const int saveCooldown = 3;
    const int soundCooldown = 2;

    while (true) {

        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(7,7), 0);

        // Adaptive background update
        accumulateWeighted(gray, backgroundFloat, learningRate);
        backgroundFloat.convertTo(background, CV_8U);

        // Robust absolute difference
        Mat diff;
        absdiff(background, gray, diff);

        // Strong threshold (ignore weak motion)
        Mat mask;
        threshold(diff, mask, pixelThreshold, 255, THRESH_BINARY);

        // Heavy noise cleanup
        morphologyEx(mask, mask, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(9,9)));

        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(5,5)));

        dilate(mask, mask,
               getStructuringElement(MORPH_RECT, Size(7,7)));

        int motionPixels = countNonZero(mask);
        bool alarm = motionPixels > alarmThreshold;

        // Contours
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool objectDetected = false;

        for (auto &c : contours) {

            double area = contourArea(c);
            if (area < minContourArea) continue;

            Rect box = boundingRect(c);

            rectangle(frame, box, Scalar(0,255,0), 3);
            objectDetected = true;
        }

        // Alarm logic
        if (alarm && objectDetected) {

            putText(frame, "ALARM! INTRUSION DETECTED",
                    Point(30, 50),
                    FONT_HERSHEY_SIMPLEX,
                    1.2,
                    Scalar(0, 0, 255),
                    3);

            time_t now = time(0);

            // 🔊 Sound cooldown
            if (difftime(now, lastSoundTime) >= soundCooldown) {
                thread(playAlarmSound).detach();
                lastSoundTime = now;
            }

            // 💾 Save cooldown
            if (difftime(now, lastSaveTime) >= saveCooldown) {

                string filename =
                    "intrusion_" + getTimestamp() + ".jpg";

                imwrite(filename, frame);
                cout << "Saved: " << filename << endl;

                lastSaveTime = now;
            }
        }

        imshow("Live Feed", frame);
        imshow("Motion Mask", mask);

        char key = waitKey(30);
        if (key == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}