#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

using cv::Mat;
using std::string;
using std::vector;


// TODO: Make type double and option

std::string type2str(int type)
{
    /*
     * Taken from
     * https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
     */

    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat getScribbleMask(const cv::Mat& image, const cv::Mat& scribbles, float eps = 1)
{
    cv::Mat diff;
    cv::absdiff(image, scribbles, diff);

    cv::Mat mask;
    cv::inRange(diff, cv::Scalar(eps, eps, eps), cv::Scalar(255, 255, 255), mask);

    return mask;
}

inline double weight(const Mat& Y, int i, int j, int m, int n, double sd = 1.0)
{
    double x = Y.at<double>(i, j);
    double y = Y.at<double>(m, n);
    double d = (x - y) * (x - y) / (2 * sd * sd);
    return std::exp(- d);
}

inline int to1Dindex(const int i, const int j, const int ncols)
{
    return i * ncols + j;
}

template <typename T>
void setupProblem(const cv::Mat& Y, const cv::Mat& scribbles,
    const cv::Mat& mask,
    Eigen::SparseMatrix<T>& AU,
    Eigen::SparseMatrix<T>& AV,
    Eigen::VectorXd& bu,
    Eigen::VectorXd& bv)
{
    typedef Eigen::Triplet<double> TD;

    auto nrows = Y.rows;
    auto ncols = Y.cols;
    auto N = nrows * ncols;

    // Handle borders

    // Inside of image
    std::vector<TD> au;
    std::vector<TD> av;
    au.reserve(N * 3);
    av.reserve(N * 3);


    // bu.resize(N);

    // Iterate in col-major order
    for (auto i = 0; i < ncols; ++i) 
    {
        for (auto j = 0; j < nrows; ++j) 
        {
            auto r = to1Dindex(i, j, ncols);

            if (mask.at<bool>(i, j)) {


            } else {
                // Set Lrr = 1
                au.push_back(TD(i, j, 1));
                av.push_back(TD(i, j, 1));
            }

            // For each of the 4 neighbors of r, compute weights
            for (int dx = -1; dx < 2; dx += 2) {
                for (int dy = -1; dy < 2; dy += 2) {
                    int m = i + dx;
                    int n = j + dy;



                }
            }
        }
    }

    AU.setFromTriplets(au.begin(), au.end());
    AV.setFromTriplets(av.begin(), av.end());
}

cv::Mat colorize(const cv::Mat& image, const cv::Mat& scribbles)
{
    cv::Mat yuv_image;
    cv::cvtColor(image, yuv_image, cv::COLOR_BGR2YUV);
    yuv_image.convertTo(yuv_image, CV_64F);

    cv::Mat yuv_marks;
    cv::cvtColor(scribbles, yuv_marks, cv::COLOR_BGR2YUV);
    yuv_marks.convertTo(yuv_marks, CV_64F);

    std::cout << type2str(yuv_image.type()) << std::endl;
    std::cout << type2str(yuv_marks.type()) << std::endl;

    cv::Mat mask = getScribbleMask(image, scribbles);

    std::vector<cv::Mat> channels;
    cv::split(yuv_image, channels);
    cv::Mat Y = channels[0];

    double minVal, maxVal;
    cv::minMaxLoc(yuv_image, &minVal, &maxVal);
    std::cout << "minVal: " << minVal << " maxVal: " << maxVal << std::endl;
    cv::minMaxLoc(yuv_marks, &minVal, &maxVal);
    std::cout << "minVal: " << minVal << " maxVal: " << maxVal << std::endl;

    // Set up matrices for U and V channels
    const int N = Y.rows * Y.cols;
    Eigen::SparseMatrix<double> AU(N, N);
    Eigen::SparseMatrix<double> AV(N, N);
    // Eigen::Matrix<double, N, 1, N, 1> bu;
    Eigen::VectorXd bu;
    Eigen::VectorXd bv;
    setupProblem<double>(Y, scribbles, mask, AU, AV, bu, bv);

    // Solve for U, V channels
    // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > solver;
    // solver.compute(AU);
    // Eigen::VectorXd U = solver.solve(bu);
    // if (solver.info() != Eigen::Success)
    // {
    //     throw std::runtime_error("Failed to solve for U channel.");
    // }
    //
    // solver.compute(AV);
    // Eigen::VectorXd V = solver.solve(bv);
    // if (solver.info() != Eigen::Success)
    // {
    //     throw std::runtime_error("Failed to solve for V channel.");
    // }

    // Combine U, V with yuv_image's Y
    // Convert U to OpenCV Mat and save in channels[1]
    // Convert V to OpenCV Mat and save in channels[2]

    // cv::Mat color_image;
    // cv::merge(channels, color_image);

    Mat YY;
    Y.convertTo(YY, CV_8U);
    cv::imshow("Y", YY);

    return mask;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cout << "Prog: ./colorize <image> <scribbles>" << std::endl;
        return 1;
    }

    try {
        cv::Mat image = cv::imread(argv[1]);
        cv::Mat scribbles = cv::imread(argv[2]);
        assert(image.size() == scribbles.size());

        cv::Mat result = colorize(image, scribbles);
        std::cout << result.size() << " " << result.channels() << std::endl;

        cv::Mat marks = cv::Mat::zeros(image.size(), CV_8UC3);
        scribbles.copyTo(marks, result);

        cv::imshow("mask", result);
        cv::imshow("Marks", marks);

        cv::waitKey();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
