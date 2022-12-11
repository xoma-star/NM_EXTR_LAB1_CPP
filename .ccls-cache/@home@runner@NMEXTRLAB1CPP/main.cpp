#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <fstream>

int n = 10;
int m = n + 2;
std::vector<std::vector<double>> A;
std::vector<double> u;
std::vector<double> f;
std::vector<std::vector<double>> A_;
std::vector<double> f_;
double h;
double sigma;
double lambda_delta;

void PrintVector(const std::vector<double>& v, const std::string& label = "") {
    if (!label.empty()) {
        std::cout << label << std::endl;
    }
    for (const auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

void PrintMatrix(const std::vector<std::vector<double>>& m, const std::string& label = "") {
    if (!label.empty()) {
        std::cout << label << std::endl;
    }
    for (const auto& x : m) {
        for (const auto& y : x) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }
}

double Random(int a, int b) {
    double f = (double)rand() / RAND_MAX;
    return a + f * (b - a);
}

double MatrixNorm(const std::vector<std::vector<double>>& m) {
    double sum = 0.;
    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m[0].size(); ++j) {
            sum += m[i][j] * m[i][j];
        }
    }
    return sqrt(sum);
}

double VectorNorm(const std::vector<double>& v) {
    double sum = 0.;
    for (int i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

std::vector<std::vector<double>> MatrixSubtractMatrix(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    std::vector<std::vector<double>> difference;
    difference.resize(m1.size());
    for (int i = 0; i < m1.size(); ++i) {
        difference[i].resize(m1[i].size());
        for (int j = 0; j < m1[i].size(); ++j) {
            difference[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return difference;
}

std::vector<double> VectorSubtractVector(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> difference;
    difference.resize(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        difference[i] = v1[i] - v2[i];
    }
    return difference;
}

std::vector<double> VectorMultiplyNumber(const std::vector<double>& v, double alpha) {
    std::vector<double> product;
    product.resize(v.size());
    for (int i = 0; i < v.size(); ++i) {
        product[i] = v[i] * alpha;
    }
    return product;
}

std::vector<double> GradientDescent(const std::function<double(std::vector<double>)>& f, const std::function<std::vector<double>(std::vector<double>)>& grad_f, const std::vector<double>& x_0, double eps, int M) {
    int k = 0;
    std::vector<double> x_prev = x_0;
    std::vector<double> x = x_prev;
    double t_k = 3;
    while (true) {
        k += 1;
        std::vector<double> dd = VectorMultiplyNumber(grad_f(x_prev), t_k);
        std::vector<double> dd2 = VectorSubtractVector(x_prev, dd);
        x = VectorSubtractVector(x_prev, VectorMultiplyNumber(grad_f(x_prev), t_k));
        if (f(x) - f(x_prev) < 0) {
            std::vector<double> x_minus_x_prev = VectorSubtractVector(x, x_prev);
            if (VectorNorm(x_minus_x_prev) <= eps || k + 1 > M) {
                return x;
            }
        }
        else {
            t_k /= 2.;
        }
        for (int i = 0; i < n; ++i) {
            x_prev[i] = x[i];
        }
    }
}

double L(const std::vector<double>& u) {
    double buff0 = 0;
    double buff1 = 0;
    double buff2 = 0;
    for (int i = 0; i < m; ++i) {
        buff0 = 0;
        for (int j = 0; j < n; ++j) {
            buff0 += A_[i][j] * u[j];
        }
        buff0 -= f_[i];
        buff1 += buff0 * buff0;
    }
    for (int i = 0; i < n; ++i) {
        buff2 += u[i] * u[i];
    }
    return sqrt(buff1) + h * sqrt(buff2) + sigma;
}

double dL(const std::vector<double>& u, int k) {
    double buff0 = 0;
    double buff1 = 0;
    double buff2 = 0;
    double buff3 = 0;
    for (int i = 0; i < m; ++i) {
        buff0 = 0;
        for (int j = 0; j < n; ++j) {
            buff0 += A_[i][j] * u[j];
        }
        buff0 -= f_[i];
        buff1 += A_[i][k] * buff0;
        buff2 += buff0 * buff0;
    }
    for (int i = 0; i < n; ++i) {
        buff3 += u[i] * u[i];
    }
    return buff1 / sqrt(buff2) + h * u[k] / sqrt(buff3);
}

std::vector<double> grad_L(const std::vector<double>& u) {
    std::vector<double> grad;
    grad.resize(n);
    for (int i = 0; i < n; ++i) {
        grad[i] = dL(u, i);
    }
    return grad;
}

std::vector<std::vector<double>> TransposeMatrix(const std::vector<std::vector<double>>& m) {
    std::vector<std::vector<double>> transposed_m;
    transposed_m.resize(m[0].size());
    for (int i = 0; i < m[0].size(); ++i) {
        transposed_m[i].resize(m.size());
    }
    for (int i = 0; i < m[0].size(); ++i) {
        for (int j = 0; j < m.size(); ++j) {
            transposed_m[i][j] = m[j][i];
        }
    }
    return transposed_m;
}

std::vector<std::vector<double>> MatrixMultiplyMatrix(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    std::vector<std::vector<double>> product;
    product.resize(m1.size());
    for (int i = 0; i < m1.size(); ++i) {
        product[i].resize(m2[1].size());
    }
    for (int i = 0; i < m1.size(); ++i) {
        for (int j = 0; j < m2[1].size(); ++j) {
            for (int k = 0; k < m2.size(); ++k) {
                product[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    return product;
}

std::vector<std::vector<double>> MatrixAddMatrix(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    std::vector<std::vector<double>> sum;
    sum.resize(m1.size());
    for (int i = 0; i < m1.size(); ++i) {
        sum[i].resize(m1[i].size());
        for (int j = 0; j < m1[i].size(); ++j) {
            sum[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return sum;
}

std::vector<std::vector<double>> MatrixMultiplyNumber(const std::vector<std::vector<double>>& m, double alpha) {
    std::vector<std::vector<double>> product;
    product.resize(m.size());
    for (int i = 0; i < m.size(); ++i) {
        product[i].resize(m[i].size());
        for (int j = 0; j < m[i].size(); ++j) {
            product[i][j] = m[i][j] * alpha;
        }
    }
    return product;
}

std::vector<std::vector<double>> GetUnitMatrix(int n) {
    std::vector<std::vector<double>> unit_matrix;
    unit_matrix.resize(n);
    for (int i = 0; i < n; ++i) {
        unit_matrix[i].resize(n);
        unit_matrix[i][i] = 1;
    }
    return unit_matrix;
}

std::vector<double> MatrixMultiplyVector(const std::vector<std::vector<double>>& m, const std::vector<double>& v) {
    std::vector<double> product;
    product.resize(m.size());
    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            product[i] += m[i][j] * v[j];
        }
    }
    return product;
}

double VectorMultiplyVector(const std::vector<double>& v1, const std::vector<double>& v2) {
    double product = 0;
    for (int i = 0; i < v1.size(); ++i) {
        product += v1[i] * v2[i];
    }
    return product;
}


std::vector<double> CGM(std::vector<std::vector<double>> A, std::vector<double> b, double eps) {
    int n = A.size();
    std::vector<double> x_prev;
    x_prev.resize(n);
    std::vector<double> x = x_prev;
    std::vector<double> r_prev = VectorSubtractVector(b, MatrixMultiplyVector(A, x_prev));
    std::vector<double> r = r_prev;
    std::vector<double> z_prev = r_prev;
    std::vector<double> z = z_prev;
    double error = 0;
    do {
        error = 0;
        double alpha_k = VectorMultiplyVector(r_prev, r_prev) / VectorMultiplyVector(MatrixMultiplyVector(A, z_prev), z_prev);
        for (int i = 0; i < n; ++i) {
            x[i] = x_prev[i] + alpha_k * z_prev[i];
            r[i] = r_prev[i] - alpha_k * MatrixMultiplyVector(A, z_prev)[i];
        }
        double beta_k = VectorMultiplyVector(r, r) / VectorMultiplyVector(r_prev, r_prev);
        for (int i = 0; i < n; ++i) {
            z[i] = r[i] + beta_k * z_prev[i];
        }
        error = VectorNorm(VectorSubtractVector(x, x_prev));
        for (int i = 0; i < n; ++i) {
            x_prev[i] = x[i];
            r_prev[i] = r[i];
            z_prev[i] = z[i];
        }
    } while (error > eps);
    return x;
}

double rho(const std::vector<double>& u, double lambda_delta) {
    double buff0 = 0;
    double buff1 = 0;
    double buff2 = 0;
    for (int i = 0; i < m; ++i) {
        buff0 = 0;
        for (int j = 0; j < n; ++j) {
            buff0 += A_[i][j] * u[j];
        }
        buff0 -= f_[i];
        buff1 += buff0 * buff0;
    }
    for (int i = 0; i < n; ++i) {
        buff2 += u[i] * u[i];
    }
    return buff1 - (lambda_delta + sigma + h * sqrt(buff2)) * (lambda_delta + sigma + h * sqrt(buff2));
}


void Generate() {
    A.resize(n);
    for (int i = 0; i < n; ++i) {
        A[i].resize(n);
        for (int j = 0; j < n; ++j) {
            A[i][j] = Random(0, 1);
        }
    }
    u.resize(n);
    for (int i = 0; i < n; ++i) {
        u[i] = Random(0, 1);
    }
    f.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i] += A[i][j] * u[j];
        }
    }
    A.resize(m);
    f.resize(m);
    u.resize(m);
    for (int i = n; i < m; ++i) {
        A[i].resize(n);
    }
    for (int j = 0; j < n; ++j) {
        A[n][j] = A[0][j] + A[n - 1][j];
        A[n + 1][j] = A[1][j] + A[n - 2][j];
    }
    f[n] = f[0] + f[n - 1];
    f[n + 1] = f[1] + f[n - 2];
    u[n] = u[0] + u[n - 1];
    u[n + 1] = u[1] + u[n - 2];
    PrintMatrix(A, "A");
    PrintVector(f, "f");
    PrintVector(u, "u");
}

void AddErrors(double e_a, double e_h) {
    std::copy(A.begin(), A.end(), std::back_inserter(A_));
    std::copy(f.begin(), f.end(), std::back_inserter(f_));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A_[i][j] = A_[i][j] * (1 + Random(-1, 1) * e_a);
        }
        f_[i] = f_[i] * (1 + Random(-1, 1) * e_h);
    }
    PrintMatrix(A_, "A_");
    PrintVector(f_, "f_");
}

void CalculateHAndSigma() {
    std::vector<std::vector<double>> a_minus_a_ = MatrixSubtractMatrix(A, A_);
    std::vector<double> f_minus_f_ = VectorSubtractVector(f, f_);
    h = MatrixNorm(a_minus_a_);
    sigma = VectorNorm(f_minus_f_);
    PrintMatrix(a_minus_a_, "a_minus_a_");
    PrintVector(f_minus_f_, "f_minus_f_");
    std::cout << "h: " << h << std::endl;
    std::cout << "sigma: " << sigma << std::endl;
}

void FindLambdaDelta() {
    std::vector<double> x_0;
    x_0.resize(n);
    for (int i = 0; i < n; ++i) {
        x_0[i] = u[i];
    }
    std::vector<double> L_min = GradientDescent(L, grad_L, x_0, 1e-6, 1000);
    lambda_delta = L(L_min);
    PrintVector(L_min);
    std::cout << "lambda_delta = " << lambda_delta << std::endl;
}

void FindSolution() {
    std::vector<std::vector<double>> unit_matrix = GetUnitMatrix(n);
    double alpha = 1;
    std::vector<std::vector<double>> matrix = MatrixAddMatrix(MatrixMultiplyMatrix(TransposeMatrix(A_), A_), MatrixMultiplyNumber(unit_matrix, alpha));
    std::vector<double> vector = MatrixMultiplyVector(TransposeMatrix(A_), f_);
    std::vector<double> answer = CGM(matrix, vector, 1e-6);
    PrintMatrix(matrix);
    PrintVector(vector);
    std::ofstream file("rho.txt");
    double value = rho(answer, lambda_delta);
    //file << alpha << " " << value << std::endl;
    double need_alpha = 0;
    double need_value = 0;
    std::vector<double> need_answer = {};
    while (alpha >= -0.01) {
        alpha -= 0.0001;
        matrix = MatrixAddMatrix(MatrixMultiplyMatrix(TransposeMatrix(A_), A_), MatrixMultiplyNumber(unit_matrix, alpha));
        vector = MatrixMultiplyVector(TransposeMatrix(A_), f_);
        answer = CGM(matrix, vector, 1e-6);
        value = rho(answer, lambda_delta);
        if (alpha <= 0.25) {
            file << alpha << " " << value << std::endl;
        }
        if (fabs(value) < 1e-4 && alpha >= 0) {
            need_alpha = alpha;
            need_value = value;
            need_answer = answer;
        }
    }
    file.close();
    std::cout << "value: " << need_value << std::endl;
    std::cout << "alpha: " << need_alpha << std::endl;
    PrintVector(need_answer);
    std::cout << "norm answ: " << VectorNorm(need_answer) << std::endl;
}

int main() {
    srand(2);
    Generate();
    double e_a = 0.0001, e_h = 0.0001;
    AddErrors(e_a, e_h);
    CalculateHAndSigma();
    FindLambdaDelta();
    FindSolution();
    PrintVector(std::vector<double>(u.begin(), u.end() - 2));
    std::vector<double> exact;
    exact.resize(n);
    for (int i = 0; i < n; ++i) {
        exact[i] = u[i];
    }
    std::cout << "norm exact: " << VectorNorm(exact) << std::endl;
}