#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

const int GAME_WIDTH = 800;
const int GAME_HEIGHT = 600;
const int GAME_TIME = 30;
const float CLICK_TIME = 1.0;

enum GameState { MENU, PLAYING, GAMEOVER, LEADERBOARD };

struct Balloon {
    Point pos;
    int radius;
    int speed;
    bool active;
};

void overlayPNG(Mat &bg, Mat &fg, Point loc){
    if(fg.empty()) return;

    for(int y=0;y<fg.rows;y++){
        for(int x=0;x<fg.cols;x++){
            Vec4b px = fg.at<Vec4b>(y,x);
            double a = px[3]/255.0;

            int bx = x+loc.x;
            int by = y+loc.y;

            if(bx>=0 && by>=0 &&
               bx<bg.cols && by<bg.rows){
                for(int c=0;c<3;c++){
                    bg.at<Vec3b>(by,bx)[c] =
                        bg.at<Vec3b>(by,bx)[c]*(1-a)
                        + px[c]*a;
                }
            }
        }
    }
}

int main(){

    string playerName;
    cout<<"Enter Player Name: ";
    cin>>playerName;

    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;

    Mat balloonSprite =
        imread("balloon.png", IMREAD_UNCHANGED);
    if(!balloonSprite.empty())
        resize(balloonSprite,
               balloonSprite,
               Size(50,50));

    Mat avatar =
        imread("avatar.png", IMREAD_UNCHANGED);
    if(!avatar.empty())
        resize(avatar, avatar,
               Size(70,70));

    RNG rng(time(0));

    vector<Balloon> balloons;
    GameState state = MENU;
    int score = 0;

    double startTime = 0;
    double hoverStart = 0;
    bool hovering = false;

    while(true){

        Mat frame;
        cap>>frame;
        if(frame.empty()) break;

        flip(frame, frame, 1);

        Mat hsv;
        cvtColor(frame,hsv,COLOR_BGR2HSV);

        Mat m1,m2;
        inRange(hsv,
            Scalar(0,100,100),
            Scalar(10,255,255),m1);
        inRange(hsv,
            Scalar(170,100,100),
            Scalar(180,255,255),m2);

        Mat redMask = m1|m2;

        Mat kernel =
        getStructuringElement(
            MORPH_ELLIPSE, Size(5,5));

        morphologyEx(redMask,
            redMask,
            MORPH_CLOSE,kernel);
        morphologyEx(redMask,
            redMask,
            MORPH_OPEN,kernel);

        Moments M =
            moments(redMask,true);

        Point centroid(-1,-1);
        Point gameCentroid(-1,-1);

        if(M.m00>0){

            centroid =
            Point(M.m10/M.m00,
                  M.m01/M.m00);

            float sx =
            (float)GAME_WIDTH/
            frame.cols;

            float sy =
            (float)GAME_HEIGHT/
            frame.rows;

            gameCentroid =
            Point(centroid.x*sx,
                  centroid.y*sy);

            circle(frame,
                   centroid,
                   10,
                   Scalar(0,255,0),2);
        }

        Mat game(GAME_HEIGHT,
                 GAME_WIDTH,
                 CV_8UC3,
                 Scalar(30,30,60));

        auto hoverClick =
        [&](Rect btn){

            if(gameCentroid.x!=-1 &&
               btn.contains(gameCentroid)){

                if(!hovering){
                    hoverStart =
                    getTickCount();
                    hovering = true;
                }

                double t =
                (getTickCount()
                -hoverStart)
                /getTickFrequency();

                rectangle(game,
                    btn,
                    Scalar(0,255,255),3);

                return t>CLICK_TIME;
            }

            hovering=false;
            return false;
        };

        if(state==MENU){

            putText(game,
            "BALLOON POP",
            Point(200,180),
            FONT_HERSHEY_SIMPLEX,
            2,
            Scalar(255,255,255),4);

            Rect startBtn(300,280,200,80);
            rectangle(game,
                startBtn,
                Scalar(0,200,0),-1);

            putText(game,
            "START",
            Point(330,335),
            FONT_HERSHEY_SIMPLEX,
            1.2,
            Scalar(255,255,255),3);

            if(hoverClick(startBtn)){
                state=PLAYING;
                score=0;
                balloons.clear();
                startTime=getTickCount();
            }
        }

        else if(state==PLAYING){

            double elapsed =
            (getTickCount()-startTime)
            /getTickFrequency();

            if(elapsed>GAME_TIME){

                ofstream file(
                    "scores.txt",
                    ios::app);
                file<<playerName
                    <<" "<<score<<endl;
                file.close();

                state=GAMEOVER;
            }

            if(rng.uniform(0,100)<4){
                Balloon b;
                b.pos=
                Point(
                rng.uniform(40,
                GAME_WIDTH-40),
                GAME_HEIGHT);
                b.radius=25;
                b.speed=
                rng.uniform(2,5);
                b.active=true;
                balloons.push_back(b);
            }

            for(auto &b:balloons){

                if(!b.active) continue;
                b.pos.y-=b.speed;

                if(!balloonSprite.empty())
                    overlayPNG(
                    game,
                    balloonSprite,
                    Point(
                    b.pos.x-25,
                    b.pos.y-25));
                else
                    circle(game,
                    b.pos,
                    b.radius,
                    Scalar(0,0,255),
                    -1);

                if(gameCentroid.x!=-1 &&
                   norm(gameCentroid
                   -b.pos)<b.radius){
                    b.active=false;
                    score++;
                }
            }

            if(gameCentroid.x!=-1 &&
               !avatar.empty())
                overlayPNG(
                game,
                avatar,
                Point(
                gameCentroid.x-35,
                gameCentroid.y-35));

            putText(game,
            "Score: "+to_string(score),
            Point(20,50),
            FONT_HERSHEY_SIMPLEX,
            1.2,
            Scalar(255,255,255),3);

            putText(game,
            "Time: "+
            to_string(
            GAME_TIME-
            (int)elapsed),
            Point(600,50),
            FONT_HERSHEY_SIMPLEX,
            1.2,
            Scalar(255,255,255),3);
        }

        else if(state==GAMEOVER){

            putText(game,
            "GAME OVER",
            Point(220,220),
            FONT_HERSHEY_SIMPLEX,
            2,
            Scalar(0,0,255),4);

            putText(game,
            "Score: "+
            to_string(score),
            Point(300,300),
            FONT_HERSHEY_SIMPLEX,
            1.4,
            Scalar(255,255,255),3);

            Rect lbBtn(280,360,240,80);
            rectangle(game,
            lbBtn,
            Scalar(0,150,200),-1);

            putText(game,
            "LEADERBOARD",
            Point(290,415),
            FONT_HERSHEY_SIMPLEX,
            0.9,
            Scalar(255,255,255),3);

            if(hoverClick(lbBtn))
                state=LEADERBOARD;
        }

        else if(state==LEADERBOARD){

            putText(game,
            "LEADERBOARD",
            Point(200,120),
            FONT_HERSHEY_SIMPLEX,
            1.8,
            Scalar(255,255,255),3);

            ifstream file(
                "scores.txt");

            string name;
            int sc;
            int y=200;

            while(file>>name>>sc){
                putText(game,
                name+" : "+
                to_string(sc),
                Point(260,y),
                FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(200,200,200),2);
                y+=40;
            }

            Rect backBtn(
                300,480,200,70);

            rectangle(game,
                backBtn,
                Scalar(0,200,0),-1);

            putText(game,
            "MENU",
            Point(350,530),
            FONT_HERSHEY_SIMPLEX,
            1,
            Scalar(255,255,255),3);

            if(hoverClick(backBtn))
                state=MENU;
        }

        imshow("Webcam",frame);
        imshow("Game",game);

        if(waitKey(20)==27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}