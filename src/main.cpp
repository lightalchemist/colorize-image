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

// TODO: Make type double an option

cv::Mat getScribbleMask(const cv::Mat& image, const cv::Mat& scribbles, double eps = 1, int nErosions=1)
{
    cv::Mat diff;
    cv::absdiff(image, scribbles, diff);
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    cv::Mat mask = channels[0] + channels[1] + channels[2];
    cv::threshold(mask, mask, eps, 255, cv::THRESH_BINARY);
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), nErosions);
    return mask;
}

template <typename T>
inline T squaredDifference(const std::vector<T>& X, int r, int s)
{
    return (X[r] - X[s]) * (X[r] - X[s]);
}

template <typename T>
void to1D(const cv::Mat& m, std::vector<T>& v)
{
    v.clear();
    auto nrows = m.rows;
    auto ncols = m.cols;
    v.reserve(nrows * ncols);
    // TODO: Use Eigen Map to map this OpenCV matrix to 1D vector so we don't consume
    // extra memory

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
void getNeighbours(int i, int j, int nrows, int ncols, std::vector<T>& neighbors)
{
    neighbors.clear();
    for (int dx = -1; dx < 2; dx += 1) {
        for (int dy = -1; dy < 2; dy += 1) {
            int m = i + dy;
            int n = j + dx;
            if ((dx == 0 && dy == 0) || m < 0 || n < 0 || m >= nrows || n >= ncols)
                continue;

            T s = m * ncols + n;
            neighbors.push_back(s);
        }
    }
}

template <typename Ti, typename Tw>
inline void getWeights(const std::vector<Tw>& values,
                        Ti r, 
                        const std::vector<Ti>& neighbors,
                        std::vector<Tw>& neighborsWeights,
                        Tw gamma) {

    neighborsWeights.clear();
    std::vector<Tw> neighborsValues;
    neighborsValues.reserve(neighbors.size() + 1);

    for (auto s : neighbors) {
        neighborsWeights.push_back(squaredDifference(values, r, s));
        neighborsValues.push_back(values[s]);
    }
    neighborsValues.push_back(values[r]);

    Tw var = variance(neighborsValues);
    Tw normalizer = 0.0;
    for (auto& w : neighborsWeights) {
        w = std::exp(- gamma * w / (2 * var));
        normalizer += w;
    }

    for (auto& w : neighborsWeights) {
        w /= normalizer;
        assert(w >= 0);
    }
}

void setupProblem(const cv::Mat& Y, const cv::Mat& scribbles, const cv::Mat& mask, 
                  Eigen::SparseMatrix<double>& A, Eigen::VectorXd& bu,
                  Eigen::VectorXd& bv, double gamma)
{
    typedef Eigen::Triplet<double> TD;

    auto nrows = Y.rows;
    auto ncols = Y.cols;
    auto nPixels = nrows * ncols;
    A.resize(nPixels, nPixels);

    std::vector<TD> coefficients;
    coefficients.reserve(nPixels * 3);
    bu.resize(nPixels);
    bv.resize(nPixels);
    bu.setZero();
    bv.setZero();

#ifndef NDEBUG
    std::cout << "bu rows: " << bu.rows() << " cols: " << bu.cols() << std::endl;
    std::cout << "bv rows: " << bv.rows() << " cols: " << bv.cols() << std::endl;
#endif

    cv::Mat yuvScribbles;
    cv::cvtColor(scribbles, yuvScribbles, cv::COLOR_BGR2YUV);
    yuvScribbles.convertTo(yuvScribbles, CV_64FC3);
    std::vector<cv::Mat> channels;
    cv::split(yuvScribbles, channels);
    cv::Mat& U = channels[1];
    cv::Mat& V = channels[2];

    // TODO: See if we can do this more efficiently using matrix reshape
    std::vector<double> y, u, v;
    std::vector<bool> hasColor;
    to1D(Y, y);
    to1D(U, u);
    to1D(V, v);
    to1D(mask, hasColor);

    const int numNeighbors = 8;
    std::vector<double> weights;
    weights.reserve(numNeighbors);
    std::vector<unsigned long> neighbors;
    neighbors.reserve(numNeighbors);
    for (auto i = 0; i < nrows; ++i) {
        for (auto j = 0; j < ncols; ++j) {
            unsigned long r = i * ncols + j;
            getNeighbours(i, j, nrows, ncols, neighbors);
            getWeights(y, r, neighbors, weights, gamma);
            coefficients.push_back(TD(r, r, 1));
            for (auto k = 0u; k < neighbors.size(); ++k) {
                auto s = neighbors[k];
                auto w = weights[k];
                if (hasColor[s]) {
                    // Move value to RHS of Ax = b
                    bu(r) += w * u[s];
                    bv(r) += w * v[s];
                } else {
                    coefficients.push_back(TD(r, s, -w));
                }
            }
        }
    }

    A.setFromTriplets(coefficients.begin(), coefficients.end());
}

cv::Mat eigen2opencv(Eigen::VectorXd& v, int nrows, int ncols) {
    cv::Mat X(nrows, ncols, CV_64FC1, v.data());
    return X;
}

cv::Mat colorize(const cv::Mat& image, const cv::Mat& scribbles, const cv::Mat& mask,
                 double gamma=2.0)
{
    cv::Mat yuvImage;
    cv::cvtColor(image, yuvImage, cv::COLOR_BGR2YUV);
    yuvImage.convertTo(yuvImage, CV_64FC3);

    std::vector<cv::Mat> channels;
    cv::split(yuvImage, channels);
    cv::Mat Y = channels[0];

    // Set up matrices for U and V channels
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd bu;
    Eigen::VectorXd bv;
    setupProblem(Y, scribbles, mask, A, bu, bv, gamma);

    // Solve for U, V channels
    std::cout << "Solving for U channel." << std::endl;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,
                    Eigen::DiagonalPreconditioner<double> > solver;

    solver.compute(A);
    Eigen::VectorXd U = solver.solve(bu);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to solve for U channel.");
    }

    std::cout << "Solving for V channel." << std::endl;
    Eigen::VectorXd V = solver.solve(bv);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to solve for V channel.");
    }

    std::cout << "Finished coloring" << std::endl;

    const int nrows = Y.rows;
    const int ncols = Y.cols;
    cv::Mat mU = eigen2opencv(U, nrows, ncols);
    cv::Mat mV = eigen2opencv(V, nrows, ncols);
    channels[1] = mU;
    channels[2] = mV;

    cv::Mat colorImage;
    cv::merge(channels, colorImage);
    colorImage.convertTo(colorImage, CV_8UC3);
    cv::cvtColor(colorImage, colorImage, cv::COLOR_YUV2BGR);

    // For debugging purposes
    cv::Mat scribblesOnly = cv::Mat::zeros(image.size(), CV_8UC3);
    scribbles.copyTo(scribblesOnly, mask);
    cv::imshow("scribbles only", scribblesOnly);
    cv::Mat YY;
    Y.convertTo(YY, CV_8U);
    cv::imshow("Y", YY);
    cv::Mat UU;
    mU.convertTo(UU, CV_8UC1);
    cv::Mat VV;
    mV.convertTo(VV, CV_8UC1);
    cv::imshow("U", UU);
    cv::imshow("V", VV);

    return colorImage;
}

int main(int argc, char* argv[])
{
    try {

        if (argc < 4) {
            std::cerr << argv[0] << " <image> <scribbles> <output> [gamma] [threshold]" << std::endl;
            return 0;
        }

        double gamma = 2.0;
        if (argc >= 5) {
            gamma = std::stod(argv[4]);
        }

        int threshold = 10;
        if (argc >= 6) {
            threshold = std::stoi(argv[5]);
        }

        Eigen::setNbThreads(2);

        std::string imagePath{argv[1]};
        std::string scribblesPath{argv[2]};
        std::string outputPath{argv[3]};

        cv::Mat image = cv::imread(imagePath);
        cv::Mat scribbles = cv::imread(scribblesPath);
        if (image.empty()) {
            std::cerr << "Failed to read file from " << imagePath << std::endl;
            return 0;
        }

        if (scribbles.empty()) {
            std::cerr << "Failed to read file from " << scribblesPath << std::endl;
            return 0;
        }

        assert(image.size() == scribbles.size());

        cv::Mat mask = getScribbleMask(image, scribbles, threshold);
        cv::Mat colorImage = colorize(image, scribbles, mask, gamma);
        cv::imwrite(outputPath, colorImage);
        cv::imshow("scribbles", scribbles);
        cv::imshow("original", image);
        cv::imshow("color", colorImage);
        cv::waitKey();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
