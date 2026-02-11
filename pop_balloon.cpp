// Balloon popper game using red-object centroid as a pointer
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

struct Balloon {
  Point2f pos;
  float radius;
  Scalar color;
  float speed; // pixels per frame
  bool alive;
};

int main(){
  VideoCapture cap(0);
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // Ask for player name before starting
  string playerName;
  cout << "Enter player name: ";
  getline(cin, playerName);
  if(playerName.empty()) playerName = "Player";

  // RNG for balloon properties
  std::mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
  uniform_real_distribution<float> distX(0.0f, 1.0f);
  // larger balloons
  uniform_real_distribution<float> distR(45.0f, 80.0f);
  uniform_real_distribution<float> distS(2.0f, 5.0f);

  vector<Balloon> balloons;
  int score = 0;

  // Timing and spawn control
  auto startTime = chrono::steady_clock::now();
  auto lastSpawn = startTime;
  const double gameDuration = 30.0; // seconds
  // spawn less frequently and cap simultaneous balloons
  const double spawnInterval = 1.5; // seconds
  const int maxSimultaneous = 5;

  Mat frame;

  while(true){
    cap >> frame;
    if(frame.empty()) break;

    // Flip frame horizontally so left/right movements match physical movements
    flip(frame, frame, 1);

    // Time calculations
    auto now = chrono::steady_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(now - startTime).count();
    double sinceSpawn = chrono::duration_cast<chrono::duration<double>>(now - lastSpawn).count();

    // Stop spawning/moving after duration
    bool running = (elapsed < gameDuration);

    // Spawn new balloon
    // only spawn if under the max simultaneous balloons
    int aliveCount = 0;
    for(const auto &bb : balloons) if(bb.alive) aliveCount++;
    if(running && sinceSpawn >= spawnInterval && aliveCount < maxSimultaneous){
      Balloon b;
      float r = distR(rng);
      b.radius = r;
      int w = frame.cols;
      b.pos.x = distX(rng) * (w - 2*r) + r;
      b.pos.y = frame.rows + r; // start below screen
      b.speed = distS(rng);
      // random color choices
      int cidx = rng() % 6;
      switch(cidx){
        case 0: b.color = Scalar(0,0,255); break; // red
        case 1: b.color = Scalar(0,255,0); break; // green
        case 2: b.color = Scalar(255,0,0); break; // blue
        case 3: b.color = Scalar(0,255,255); break; // yellow
        case 4: b.color = Scalar(255,0,255); break; // magenta
        default: b.color = Scalar(255,255,0); break; // cyan
      }
      b.alive = true;
      balloons.push_back(b);
      lastSpawn = now;
    }

    // Detect red centroid (player pointer)
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    Scalar lowerRed1(0, 100, 100), upperRed1(10, 255, 255);
    Scalar lowerRed2(170, 100, 100), upperRed2(180, 255, 255);
    Mat m1, m2;
    inRange(hsv, lowerRed1, upperRed1, m1);
    inRange(hsv, lowerRed2, upperRed2, m2);
    Mat redMask = m1 | m2;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
    morphologyEx(redMask, redMask, MORPH_OPEN, kernel);
    Moments mo = moments(redMask, true);
    Point2f centroid(-1,-1);
    if(mo.m00 > 1000){ // threshold to avoid noise
      centroid.x = float(mo.m10 / mo.m00);
      centroid.y = float(mo.m01 / mo.m00);
    }

    // Update balloons (move up) and check collisions
    for(auto &b : balloons){
      if(!b.alive) continue;
      if(running) b.pos.y -= b.speed; // move upward while running

      // If balloon fully above screen, mark as dead (escaped)
      if(b.pos.y + b.radius < 0){
        b.alive = false;
        continue;
      }

      // Collision check with centroid
      if(centroid.x >= 0){
        float dx = centroid.x - b.pos.x;
        float dy = centroid.y - b.pos.y;
        float dist2 = dx*dx + dy*dy;
        float rsum = b.radius + 12.0f; // 12 px tolerance for pointer
        if(dist2 <= rsum*rsum){
          b.alive = false; // popped
          score += 1;
        }
      }
    }

    // Create gradient landscape background instead of showing webcam image
    Mat out(frame.size(), frame.type());
    int rows = out.rows;
    int cols = out.cols;
    // Sky gradient (top) -> from light blue to deeper blue
    Vec3b skyTop = Vec3b(200, 220, 255);
    Vec3b skyBottom = Vec3b(100, 160, 255);
    // Ground gradient (bottom) -> light green to darker green
    Vec3b groundTop = Vec3b(100, 200, 100);
    Vec3b groundBottom = Vec3b(40, 120, 40);
    for(int y=0;y<rows;y++){
      Vec3b color;
      if(y < int(rows*0.65)){
        float t = float(y) / float(max(1, int(rows*0.65)));
        for(int c=0;c<3;c++) color[c] = uchar(skyTop[c] * (1-t) + skyBottom[c] * t);
      } else {
        float t = float(y - int(rows*0.65)) / float(max(1, rows - int(rows*0.65)));
        for(int c=0;c<3;c++) color[c] = uchar(groundTop[c] * (1-t) + groundBottom[c] * t);
      }
      for(int x=0;x<cols;x++) out.at<Vec3b>(y,x) = color;
    }
    for(const auto &b : balloons){
      if(!b.alive) continue;
      circle(out, b.pos, int(b.radius), b.color, -1, LINE_AA);
      // small black border
      circle(out, b.pos, int(b.radius), Scalar(0,0,0), 2, LINE_AA);
    }

    // Draw centroid pointer (black)
    if(centroid.x >= 0){
      circle(out, centroid, 12, Scalar(0,0,0), 2, LINE_AA);
      circle(out, centroid, 3, Scalar(0,0,0), -1, LINE_AA);
    }

    // HUD: score and time left (bold format)
    int timeLeft = int(max(0.0, gameDuration - elapsed));
    string scoreText = "Score: " + to_string(score);
    string timeText = "Time: " + to_string(timeLeft) + "s";
    putText(out, scoreText, Point(10,30), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255,255,255), 4);
    putText(out, timeText, Point(10,80), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(255,255,255), 4);

    // Create split-screen: webcam on left, game on right
    Mat splitWindow(frame.rows, frame.cols * 2, frame.type());
    Rect leftROI(0, 0, frame.cols, frame.rows);
    Rect rightROI(frame.cols, 0, frame.cols, frame.rows);
    frame.copyTo(splitWindow(leftROI));
    out.copyTo(splitWindow(rightROI));

    imshow("Pop Balloons", splitWindow);

    // Stop condition after time finishes: display final score and wait for key
    if(!running){
      // show final score overlay
      Mat final = out.clone();
      string finalText = "Time's up! Final Score: " + to_string(score);
      int baseline = 0;
      Size tsize = getTextSize(finalText, FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseline);
      Point center((final.cols - tsize.width)/2, (final.rows - tsize.height)/2);
      rectangle(final, Rect(center.x-10, center.y-40, tsize.width+20, tsize.height+60), Scalar(0,0,0), -1);
      putText(final, finalText, Point(center.x, center.y), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0,255,0), 3);
      imshow("Pop Balloons", final);
      
      // Append score to scores.txt with timestamp and player name
      ofstream ofs("scores.txt", ios::app);
      if(ofs){
        time_t nowt = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&nowt));
        ofs << buf << "\t" << playerName << "\t" << score << "\n";
        ofs.close();
      } else {
        cerr << "Warning: could not open scores.txt for writing." << endl;
      }
      // wait for 5 seconds or key press
      int k = waitKey(5000);
      break;
    }

    // frame rate wait
    int key = waitKey(25);
    if(key == 27) break; // ESC to quit
  }

  cap.release();
  destroyAllWindows();
  return 0;
}