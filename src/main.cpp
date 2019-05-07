#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "colorize.hpp"


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

        // Eigen::setNbThreads(2);

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
