#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>


#include "ndarray_converter.h"
using namespace cv;
using namespace std;


namespace py = pybind11;

// void show_image(cv::Mat image) {
//   cv::imshow("image_from_Cpp", image);
//   cv::waitKey(0);
// }

// cv::Mat read_image(std::string image_name) {
// #if CV_MAJOR_VERSION < 4
//   cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
// #else
//   cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
// #endif
//   return image;
// }

Mat dahu_scribble(Mat image, Mat distri_gray, Mat F_fg)
{

    // cout << "image channel" << " "  << image.channels() << endl;
    // cout << "image channel" << " "  << distri_gray.rows << endl;
    // cout << "image channel" << " "  << F_fg.rows << endl;

    // imwrite("image.png", image);
    // imwrite("distri_gray.png", distri_gray);
    // imwrite("F_fg.png", F_fg);

    int nRows = distri_gray.rows;
    int nCols = distri_gray.cols;

    // Merge 2 images

    Mat ima_merge;

    vector<Mat> channels;
    channels.push_back(image);
    channels.push_back(distri_gray);
    merge(channels, ima_merge);



    // Immerse
    Mat U_lower(nRows*2-1, nCols*2-1, CV_8UC4, Scalar(0,0,0,0));
    Mat U_upper(nRows*2-1, nCols*2-1, CV_8UC4, Scalar(0,0,0,0));

    int height = nRows*2-1;
    int width = nCols*2-1;

    for (int i = 0; i < nCols; i++) {
        for (int j = 0; j < nRows; j++) {

            Point2f p = Point2f(i,j);
            Vec4b a = ima_merge.at<Vec4b>(p);
            Vec4b b,c,d;

            Point2f p_b = p + Point2f(0,1);
            if (p_b.x >= 0 and p_b.x < nCols and p_b.y >= 0 and p_b.y < nRows)
                b = ima_merge.at<Vec4b>(p_b);

            Point2f p_c = p + Point2f(1,0);
            if (p_c.x >= 0 and p_c.x < nCols and p_c.y >= 0 and p_c.y < nRows)
                c = ima_merge.at<Vec4b>(p_c);

            Point2f p_d = p + Point2f(1,1);
            if (p_d.x >= 0 and p_d.x < nCols and p_d.y >= 0 and p_d.y < nRows)
                d = ima_merge.at<Vec4b>(p_d);



            Point2f q = 2 * p;
            U_lower.at<Vec4b>(q) = ima_merge.at<Vec4b>(p);
            U_upper.at<Vec4b>(q) = ima_merge.at<Vec4b>(p);

            for (int i = 0 ; i < 4; i++)
            {
                uint8_t min1 = std::min(a[i],b[i]), min2 = std::min(a[i],c[i]);
                uint8_t max1 = std::max(a[i],b[i]), max2 = std::max(a[i],c[i]);
                uint8_t min3 = std::min(d[i], std::min(c[i], min1));
                uint8_t max3 = std::max(d[i], std::max(c[i], max1));

           

                Point2f q_b = q + Point2f(0,1);
                if (q_b.x >= 0 && q_b.x < width && q_b.y >= 0 and q_b.y < height)
                {
                    U_lower.at<Vec4b>(q_b)[i] = min1;
                    U_upper.at<Vec4b>(q_b)[i] = max1;
                }

                Point2f q_c = q + Point2f(1,0);
                if (q_c.x >= 0 && q_c.x < width && q_c.y >= 0 and q_c.y < height)
                {
                    U_lower.at<Vec4b>(q_c)[i] = min2;
                    U_upper.at<Vec4b>(q_c)[i] = max2;            
                }

                Point2f q_d = q + Point2f(1,1);
                if (q_d.x >= 0 && q_d.x < width && q_d.y >= 0 and q_d.y < height)
                {
                    U_lower.at<Vec4b>(q_d)[i] = min3;
                    U_upper.at<Vec4b>(q_d)[i] = max3;       
                }
            }

        }
    }





    Mat state = Mat::zeros(height, width, CV_8UC1);
    Mat min_image = Mat::zeros(height, width, CV_8UC4);
    Mat max_image = Mat::zeros(height, width, CV_8UC4);
    Mat dmap = Mat::zeros(height, width, CV_8UC4);
    Mat dmap_scalar = Mat::zeros(height, width, CV_8UC1);

    Mat Ub = Mat::zeros(height, width, CV_8UC4);

    // Priority queue

    vector<queue<Point2f> > Q(256*4);

    // Put seeds on the markers
    // change the state of pixels



    for (int i = 0 ; i < width ; i ++)
    {
        for (int j = 0 ; j < height ; j ++)
        {
            Point2f p = Point2f(i,j);

            if (F_fg.at<uint8_t>(p) > 100)
            {
                //cout << "True" << endl;
                state.at<uint8_t>(p) = 1;
                dmap.at<Vec4b>(p) = {0,0,0,0};
                int temp = dmap.at<Vec4b>(p)[0]+dmap.at<Vec4b>(p)[1]+dmap.at<Vec4b>(p)[2]+ dmap.at<Vec4b>(p)[3];
                Q[temp].push(p);
                Ub.at<Vec4b>(p) = U_lower.at<Vec4b>(p);
                min_image.at<Vec4b>(p) = Ub.at<Vec4b>(p);
                max_image.at<Vec4b>(p) = Ub.at<Vec4b>(p);
            }
            else
            {
                state.at<uint8_t>(p) = 0;
                dmap.at<Vec4b>(p) = {255,255,255,255};   
                // min_image.at<uint8_t>(p) = Ub.at<uint8_t>(p);
                // max_image.at<uint8_t>(p) = Ub.at<uint8_t>(p);                            
            }
        }
    }


    int dx[4] = {1 ,-1 , 0 , 0};
    int dy[4] = {0 , 0, 1, -1};

    // // proceed the propagation of the pixel from marker to all pixels

    for (int lvl = 0; lvl < 256*4 ; lvl++)
    {
        while (!Q[lvl].empty())
        {
            Point2f p = Q[lvl].front();
            Q[lvl].pop();
            Vec4b l_cur = Ub.at<Vec4b>(p);

            if (state.at<uint8_t>(p) == 2)
                continue;       

            state.at<uint8_t>(p) = 2;

            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int x  = p.x + dx[n1];
                int y  = p.y + dy[n1];  

                if (x >= 0 and x < width and y >= 0 and y < height)
                {
                    Point2f r = Point2f(x,y);
                    Vec4b l_ ;

                    for (int k = 0; k < 4; k++)
                    {
                        if (l_cur[k] < U_lower.at<Vec4b>(r)[k])
                            l_[k] = U_lower.at<Vec4b>(r)[k];
                        else if (l_cur[k] > U_upper.at<Vec4b>(r)[k])
                            l_[k] = U_upper.at<Vec4b>(r)[k];
                        else
                            l_[k] = l_cur[k];                        
                    }


                    Ub.at<Vec4b>(r) = l_;  
                    int temp_r = dmap.at<Vec4b>(r)[0]+dmap.at<Vec4b>(r)[1]+dmap.at<Vec4b>(r)[2]+dmap.at<Vec4b>(r)[3];
                    int temp_p = dmap.at<Vec4b>(p)[0]+dmap.at<Vec4b>(p)[1]+dmap.at<Vec4b>(p)[2]+dmap.at<Vec4b>(p)[3];                    

                    if (state.at<uint8_t>(r)==1 and temp_r> temp_p)
                    {
                        // if (F_fg.at<uint8_t>(r) > 100)
                        //  cout << "Dkm" << int(dmap.at<uint8_t>(r)) << endl << int(dmap.at<uint8_t>(p))  << endl;

                        min_image.at<Vec4b>(r) = min_image.at<Vec4b>(p);
                        max_image.at<Vec4b>(r) = max_image.at<Vec4b>(p);

                        for (int k = 0; k < 4; k++)
                        {
                            if (Ub.at<Vec4b>(r)[k] < min_image.at<Vec4b>(r)[k])
                                min_image.at<Vec4b>(r)[k] = Ub.at<Vec4b>(r)[k];
                            if (Ub.at<Vec4b>(r)[k] > max_image.at<Vec4b>(r)[k])
                                max_image.at<Vec4b>(r)[k] = Ub.at<Vec4b>(r)[k];
                        }

                        temp_r = dmap.at<Vec4b>(r)[0]+dmap.at<Vec4b>(r)[1]+dmap.at<Vec4b>(r)[2]+dmap.at<Vec4b>(r)[3];
                        int temp_dis = 0;
                        for (int i = 0; i < 4; i++)
                            temp_dis = temp_dis + max_image.at<Vec4b>(r)[i] - min_image.at<Vec4b>(r)[i];

                        if (temp_r > temp_dis)
                        {
                            dmap.at<Vec4b>(r) = max_image.at<Vec4b>(r) - min_image.at<Vec4b>(r);
                            Q[temp_dis].push(r);
                        } 
                    }

                    else if (state.at<uint8_t>(r) == 0)
                    {
                        min_image.at<Vec4b>(r) = min_image.at<Vec4b>(p);
                        max_image.at<Vec4b>(r) = max_image.at<Vec4b>(p);
                        
                        for (int k = 0; k < 4; k++)
                        {
                            if (Ub.at<Vec4b>(r)[k] < min_image.at<Vec4b>(r)[k])
                                min_image.at<Vec4b>(r)[k] = Ub.at<Vec4b>(r)[k];
                            if (Ub.at<Vec4b>(r)[k] > max_image.at<Vec4b>(r)[k])
                                max_image.at<Vec4b>(r)[k] = Ub.at<Vec4b>(r)[k];
                        }

                        dmap.at<Vec4b>(r) = max_image.at<Vec4b>(r) - min_image.at<Vec4b>(r);
                        int temp_dis = 0;
                        for (int i = 0; i < 4; i++)
                            temp_dis = temp_dis + max_image.at<Vec4b>(r)[i] - min_image.at<Vec4b>(r)[i]; 

                        Q[temp_dis].push(r);
                        state.at<uint8_t>(r) = 1; 

                    }

                    else
                        continue;

                }

            }
        }
    }

    imwrite("Ub.png", Ub);


    for (int i = 0 ; i < width ; i ++)
    {
        for (int j = 0 ; j < height ; j ++)
        {
            Point2f p = Point2f(i,j);
            uint8_t value = 0;
            for (int k = 0; k < 4; k++)
            {
                value = value + int(dmap.at<Vec4b>(p)[k]/4);
            }
            dmap_scalar.at<uint8_t>(p) = value;

        }
    }
    return dmap_scalar;
}


// cv::Mat passthru(cv::Mat image) {
//   return image;
// }

// cv::Mat cloneimg(cv::Mat image) {
//   return image.clone();
// }

// class AddClass {
// public:
//   AddClass(int value) : value(value) {}

//   cv::Mat add(cv::Mat input) {
//     return input + this->value;
//   }

// private:
//   int value;
// };

PYBIND11_MODULE(example, m) {

  NDArrayConverter::init_numpy();

  m.def("dahu_scribble", &dahu_scribble);

  // m.def("show_image", &show_image, "A function that show an image",
  //       py::arg("image"));

  // m.def("passthru", &passthru, "Passthru function", py::arg("image"));
  // m.def("clone", &cloneimg, "Clone function", py::arg("image"));

  // py::class_<AddClass>(m, "AddClass")
  //   .def(py::init<int>())
  //   .def("add", &AddClass::add);
}
