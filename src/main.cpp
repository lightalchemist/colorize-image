#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

// using cv::Mat;
// using std::string;
// using std::vector;

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

inline double weight(const cv::Mat& Y, int i, int j, int m, int n, double sd = 1.0)
{
    double x = Y.at<double>(i, j);
    double y = Y.at<double>(m, n);
    double d = (x - y) * (x - y) / (2 * sd * sd);
    return std::exp(-d);
}

template <typename T>
inline T squaredDifference(const std::vector<T>& X, int r, int s)
{
    return (X[r] - X[s]) * (X[r] - X[s]);
}

inline int to1Dindex(const int i, const int j, const int ncols)
{
    return i * ncols + j;
}

template <typename T>
void to1D(const cv::Mat& m, std::vector<T>& v)
{
    v.clear();
    auto nrows = m.rows;
    auto ncols = m.cols;
    v.reserve(nrows * ncols);
    // if (m.isContinuous()) {
    //     std::cout << "Convert continuous matrix" << std::endl;
    //     v.assign(m.datastart, m.dataend);
    //     assert((nrows * ncols) == v.size());
    // } else {
        int total = 0;
        for (auto i = 0; i < m.rows; ++i) {
            v.insert(v.end(), m.ptr<T>(i), m.ptr<T>(i) + ncols);
            total += ncols;
        }
        std::cout << "Added " << total << " entries" << std::endl;
    // }
}

template <typename T>
T variance(const std::vector<T>& vals)
{
    T sum = 0;
    T squaredSum = 0;
    for (auto v : vals) {
        sum += v;
        squaredSum += v * v;
    }

    assert (sum >= 0);
    assert (squaredSum >= 0);

    T n = vals.size();
    return squaredSum / n - (sum * sum) / (n * n) + 0.001;
}

template <typename T>
void get_neighbors(int i, int j, int nrows, int ncols, std::vector<T>& neighbors)
{
    neighbors.clear();
    for (int dx = -1; dx < 2; dx += 1) {
        for (int dy = -1; dy < 2; dy += 1) {
            if (dx == 0 && dy == 0)
                continue;

            int m = i + dy;
            int n = j + dx;
            if (m < 0 || n < 0 || m >= nrows || n >= ncols)
                continue;

            int s = m * ncols + n;
            neighbors.push_back(s);
        }
    }
}

// template <typename T>
void setupProblem(const cv::Mat& Y,
    const cv::Mat& scribbles,
    const cv::Mat& mask,
    Eigen::SparseMatrix<double>& A,
    Eigen::VectorXd& bu,
    Eigen::VectorXd& bv)
{
    typedef Eigen::Triplet<double> TD;

    auto nrows = Y.rows;
    auto ncols = Y.cols;
    auto N = nrows * ncols;

    // TODO: Set size of sparse matrix inside here
    std::vector<TD> a;
    a.reserve(N * 3);
    bu.resize(N);
    bv.resize(N);
    bu.setZero();
    bv.setZero();

    cv::Mat yuv_marks;
    cv::cvtColor(scribbles, yuv_marks, cv::COLOR_BGR2YUV);
    yuv_marks.convertTo(yuv_marks, CV_64F);
    std::vector<cv::Mat> channels;
    cv::split(yuv_marks, channels);
    cv::Mat& U = channels[1];
    cv::Mat& V = channels[2];

    // TODO: See if we can do this more efficiently using matrix reshape
    std::vector<double> y, u, v;
    std::vector<bool> m;
    to1D(Y, y);
    to1D(U, u);
    to1D(V, v);
    to1D(mask, m);

    std::cout << "Y size: " << Y.size() << " # channels: " << Y.channels() << std::endl;
    std::cout << "y size: " << y.size() << std::endl;
    std::cout << "u size: " << u.size() << std::endl;
    std::cout << "v size: " << v.size() << std::endl;

    const int n_neighbors = 8;
    std::vector<double> nw, ny;
    nw.reserve(n_neighbors);
    ny.reserve(n_neighbors + 1);

    std::vector<unsigned long> neighbors;
    neighbors.reserve(n_neighbors);

    std::cout << "Generating matrices" << std::endl;
    std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;
    for (auto i = 0; i < nrows; ++i) {
        for (auto j = 0; j < ncols; ++j) {
            nw.clear();
            ny.clear();
            neighbors.clear();

            get_neighbors(i, j, nrows, ncols, neighbors);
            // std::cout << "i: " << i << " j: " << j << std::endl;
            // std::cout << "Got neighbors" << std::endl;
            // std::cout << "# neighbors: " << neighbors.size() << std::endl;
            // for (auto s : neighbors) {
            //     std::cout << "s row index: " << s / ncols << " s col index: " << s % ncols << std::endl;
            // }

            auto r = i * ncols + j;

            // Compute weights for neighbors and normalize them
            for (auto s : neighbors) {
                nw.push_back(squaredDifference(y, r, s));
                ny.push_back(y[s]);
            }
            ny.push_back(y[r]);
            // std::cout << "Computed squared diff" << std::endl;

            double normalizer = 0;
            double var = variance(ny);
            // std::cout << "Variance: " << var << std::endl;
            // double var = 2.0;
            for (auto& w : nw) {
                assert(w >= 0);
                w = std::exp(-w / (2 * var));
                normalizer += w;
            }

            for (auto& w : nw) {
                w /= normalizer;
            }

            // for (auto w : nw) {
            //     std::cout << "w: " << w << std::endl;
            // }

            // For each neighbor, set the appropriate coefficient
            for (auto k = 0; k < neighbors.size(); ++k) {

                auto s = neighbors[k];
                auto wk = nw[k];
                if (m[s]) {
                    bu(r) += wk * u[s];
                    bv(r) += wk * v[s];
                } else {
                    a.push_back(TD(r, s, -wk));
                }
            }

            // Current pixel itself
            if (m[r]) {
                bu(r) -= u[r];
                bv(r) -= v[r];
            } else {
                // Set Lrr = 1
                a.push_back(TD(r, r, 1));
            }
        }
    }

    std::cout << "Setting entries of A" << std::endl;
    std::cout << "A rows: " << A.rows() << " A cols: " << A.cols() << std::endl;
    std::cout << "a size: " << a.size() << std::endl;
    
    for (auto i = 0; i < a.size() && i < 10; ++i) {
        auto tp = a[i];
        std::cout << "row: " << tp.row() << " col: " << tp.col() << " val: " << tp.value() << std::endl;
    }

    for (auto t : a) {
        if (t.row() < 0 || t.row() >= N) {
            std::cout << "Invalid row index: " << t.row() << std::endl;
        }
        if (t.col() < 0 || t.col() >= N) {
            std::cout << "Invalid col index: " << t.col() << std::endl;
        }

    }

    A.setFromTriplets(a.begin(), a.end());
}

cv::Mat colorize(const cv::Mat& image, const cv::Mat& scribbles)
{
    cv::Mat yuv_image;
    cv::cvtColor(image, yuv_image, cv::COLOR_BGR2YUV);
    yuv_image.convertTo(yuv_image, CV_64F);

    cv::Mat yuv_marks;
    cv::cvtColor(scribbles, yuv_marks, cv::COLOR_BGR2YUV);
    yuv_marks.convertTo(yuv_marks, CV_64F);

    cv::Mat mask = getScribbleMask(image, scribbles);

    std::vector<cv::Mat> channels;
    cv::split(yuv_image, channels);
    cv::Mat Y = channels[0];

    // Set up matrices for U and V channels
    // TODO: Should move resize inside setupProblem()
    const int N = Y.rows * Y.cols;
    Eigen::SparseMatrix<double> A(N, N);
    Eigen::VectorXd bu;
    Eigen::VectorXd bv;

    // setupProblem<double>(Y, scribbles, mask, A, bu, bv);
    setupProblem(Y, scribbles, mask, A, bu, bv);

    // // Solve for U, V channels
    // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > solver;
    // solver.compute(A);
    // Eigen::VectorXd U = solver.solve(bu);
    // if (solver.info() != Eigen::Success)
    // {
    //     throw std::runtime_error("Failed to solve for U channel.");
    // }
    //
    // // solver.compute(A);
    // Eigen::VectorXd V = solver.solve(bv);
    // if (solver.info() != Eigen::Success)
    // {
    //     throw std::runtime_error("Failed to solve for V channel.");
    // }

    // Combine U, V with yuv_image's Y
    // Convert U to OpenCV Mat and save in channels[1]
    // Convert V to OpenCV Mat and save in channels[2]
    // cv::Mat newU(
    // auto p = U.data();
    // cv::Mat newU(U.data()).reshape(0, nrows);
    // std::cout << typeid(U.data()).name() << std::endl;

    // cv::Mat color_image;
    // cv::merge(channels, color_image);
    // Convert to appropriate CV_8UC3, perhaps with the necessary clipping

    cv::Mat YY;
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
