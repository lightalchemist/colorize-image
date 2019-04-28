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
    // TODO: Fix this bug
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
T variance(const std::vector<T>& vals, T eps=0.01)
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
    return squaredSum / n - (sum * sum) / (n * n) + eps;
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

    std::cout << "bu rows: " << bu.rows() << " cols: " << bu.cols() << std::endl;
    std::cout << "bv rows: " << bv.rows() << " cols: " << bv.cols() << std::endl;

    cv::Mat yuv_marks;
    cv::cvtColor(scribbles, yuv_marks, cv::COLOR_BGR2YUV);
    yuv_marks.convertTo(yuv_marks, CV_64FC3);
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

            auto r = i * ncols + j;

            // Compute weights for neighbors and normalize them
            for (auto s : neighbors) {
                nw.push_back(squaredDifference(y, r, s));
                ny.push_back(y[s]);
            }
            ny.push_back(y[r]);

            double var = variance(ny, 0.1);
            // std::cout << "Variance: " << var << std::endl;
            // double var = 1.0;
            double normalizer = 0;
            double gamma = 0.8;
            for (auto& w : nw) {
                w = std::exp(- gamma * w / (2 * var));
                normalizer += w;
            }

            for (auto& w : nw) {
                w /= normalizer;
                assert(w >= 0);
            }

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
            // if (m[r]) {
            //     bu(r) -= u[r];
            //     bv(r) -= v[r];
            // } else {
                // Set A[r,r] = 1
                a.push_back(TD(r, r, 1));
            // }
        }
    }

    std::cout << "Setting entries of A" << std::endl;
    std::cout << "A rows: " << A.rows() << " A cols: " << A.cols() << std::endl;
    std::cout << "a size: " << a.size() << std::endl;

#ifndef NDEBUG
    std::cout << "Debugging" << std::endl;
    for (auto t : a) {
        if (t.row() < 0 || t.row() >= N) {
            std::cout << "Invalid row index: " << t.row() << std::endl;
        }
        if (t.col() < 0 || t.col() >= N) {
            std::cout << "Invalid col index: " << t.col() << std::endl;
        }
    }
#endif

    A.setFromTriplets(a.begin(), a.end());
}

cv::Mat eigen2opencv(Eigen::VectorXd& v, int nrows, int ncols) {
    // std::vector<double> data;
    // data.reserve(nrows * ncols);

    cv::Mat X(nrows, ncols, CV_64FC1, v.data());
    return X;
}

cv::Mat colorize(const cv::Mat& image, const cv::Mat& scribbles, const cv::Mat& mask)
{
    cv::Mat yuv_image;
    cv::cvtColor(image, yuv_image, cv::COLOR_BGR2YUV);
    yuv_image.convertTo(yuv_image, CV_64FC3);

    // cv::Mat mask = getScribbleMask(image, scribbles, 40);
    // TODO: Use image morphology to remove a think layer from this to exclude accidentally
    // included background.
    cv::imshow("mask", mask);

    cv::Mat marks = cv::Mat::zeros(image.size(), CV_8UC3);
    scribbles.copyTo(marks, mask);
    cv::imshow("Scribbles", marks);

    std::vector<cv::Mat> channels;
    cv::split(yuv_image, channels);
    cv::Mat Y = channels[0];

    // Set up matrices for U and V channels
    // TODO: Should move resize inside setupProblem()
    const int nrows = Y.rows;
    const int ncols = Y.cols;
    const int N = nrows * ncols;
    Eigen::SparseMatrix<double> A(N, N);
    Eigen::VectorXd bu;
    Eigen::VectorXd bv;

    setupProblem(Y, scribbles, mask, A, bu, bv);

    // Solve for U, V channels
    std::cout << "Solving for U channel." << std::endl;
    // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>,
    //                                      Eigen::DiagonalPreconditioner<double> > solver;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,
                    Eigen::DiagonalPreconditioner<double> > solver;

    solver.compute(A);
    Eigen::VectorXd U = solver.solve(bu);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to solve for U channel.");
    }

    cv::Mat mU = eigen2opencv(U, nrows, ncols);

    // solver.compute(A);
    std::cout << "Solving for V channel." << std::endl;
    Eigen::VectorXd V = solver.solve(bv);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to solve for V channel.");
    }

    cv::Mat mV = eigen2opencv(V, nrows, ncols);

    channels[1] = mU;
    channels[2] = mV;

    // Combine U, V with yuv_image's Y
    // std::cout << typeid(U.data()).name() << std::endl;

    cv::Mat color_image;
    cv::merge(channels, color_image);
    color_image.convertTo(color_image, CV_8UC3);
    cv::cvtColor(color_image, color_image, cv::COLOR_YUV2BGR);

    cv::Mat YY;
    Y.convertTo(YY, CV_8U);
    cv::imshow("Y", YY);

    cv::Mat UU;
    mU.convertTo(UU, CV_8UC1);
    cv::Mat VV;
    mV.convertTo(VV, CV_8UC1);
    cv::imshow("UU", UU);
    cv::imshow("VV", VV);

    return color_image;
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cout << "Prog: ./colorize <image> <scribbles> <mask> <output>" << std::endl;
        return 1;
    }


    try {
        Eigen::setNbThreads(2);

        cv::Mat image = cv::imread(argv[1]);
        cv::Mat scribbles = cv::imread(argv[2]);
        cv::Mat mask = cv::imread(argv[3]);
        assert(image.size() == scribbles.size());

        cv::Mat color_image = colorize(image, scribbles, mask);
        cv::imwrite(argv[4], color_image);

        cv::imshow("color", color_image);

        cv::waitKey();

    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
