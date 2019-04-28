
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
// #include <QImage>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);

int main(int argc, char** argv)
{
    // if (argc != 2) {
    //     std::cerr << "Error: expected one and only one argument.\n";
    //     return -1;
    // }

    int n = 2;   // size of the image
    int m = n * n; // number of unknows (=number of pixels)
    // Assembly:
    std::vector<T> coefficients; // list of non-zeros coefficients
    Eigen::MatrixXd b(m, 1);        // the right hand side-vector resulting from the constraints

    b.setRandom();
    std::cout << b << std::endl;

    // b.setZero();
    b.resize(n, n);
    // b.setZero();

    std::cout << b << std::endl;


    Eigen::VectorXd bu;
    bu.resize(6);
    bu.fill(5);
    std::cout << "bu: " << std::endl;
    std::cout << bu << std::endl;
    std::cout << "-----------------" << std::endl;

    // buildProblem(coefficients, b, n);
    // SpMat A(m, m);
    // A.setFromTriplets(coefficients.begin(), coefficients.end());
    // // Solving:
    // Eigen::SimplicialCholesky<SpMat> chol(A); // performs a Cholesky factorization of A
    // Eigen::VectorXd x = chol.solve(b);        // use the factorization to solve for the given right hand side

    // Export the result to a file:
    // saveAsBitmap(x, n, argv[1]);

    Eigen::SparseMatrix<double> A(5, 5);
    std::vector<T> entries;
    entries.push_back(T(0, 0, 1));
    entries.push_back(T(1, 1, 2));
    entries.push_back(T(2, 2, 9));
    entries.push_back(T(3, 3, 3));
    A.setFromTriplets(entries.begin(), entries.end());

    std::cout << A << std::endl;

    return 0;
}

void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
    Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
    int n = int(boundary.size());
    int id1 = i + j * n;
    if (i == -1 || i == n)
        b(id) -= w * boundary(j); // constrained coefficient
    else if (j == -1 || j == n)
        b(id) -= w * boundary(i); // constrained coefficient
    else
        coeffs.push_back(T(id, id1, w)); // unknown coefficient
}
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
    b.setZero();
    Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, M_PI).sin().pow(2);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int id = i + j * n;
            insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
            insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
            insertCoefficient(id, i, j, 4, coefficients, b, boundary);
        }
    }
}

// void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
// {
//   Eigen::Array<unsigned char,Eigen::Dynamic,Eigen::Dynamic> bits = (x*255).cast<unsigned char>();
//   QImage img(bits.data(), n,n, QImage::Format_Indexed8);
//   img.setColorCount(256);
//   for(int i=0;i<256;i++) img.setColor(i,qRgb(i,i,i));
//   img.save(filename);
// }
