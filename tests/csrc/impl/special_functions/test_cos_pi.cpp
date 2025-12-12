#include <gtest/gtest.h>

#include <torchscience/csrc/impl/special_functions/cos_pi.h>

#include <cmath>
#include <limits>

using torchscience::impl::special_functions::cos_pi;
using torchscience::impl::special_functions::cos_pi_backward;

TEST(CosPi, Integers) {
    // cos(n * pi) = (-1)^n for all integers n
    EXPECT_NEAR(cos_pi(0.0), 1.0, 1e-10);
    EXPECT_NEAR(cos_pi(1.0), -1.0, 1e-10);
    EXPECT_NEAR(cos_pi(2.0), 1.0, 1e-10);
    EXPECT_NEAR(cos_pi(-1.0), -1.0, 1e-10);
    EXPECT_NEAR(cos_pi(-2.0), 1.0, 1e-10);
}

TEST(CosPi, HalfIntegers) {
    // cos(pi/2) = 0, cos(3*pi/2) = 0
    EXPECT_NEAR(cos_pi(0.5), 0.0, 1e-10);
    EXPECT_NEAR(cos_pi(1.5), 0.0, 1e-10);
    EXPECT_NEAR(cos_pi(-0.5), 0.0, 1e-10);
    EXPECT_NEAR(cos_pi(2.5), 0.0, 1e-10);
}

TEST(CosPi, QuarterValues) {
    // cos(pi/4) = sqrt(2)/2
    double sqrt2_2 = std::sqrt(2.0) / 2.0;
    EXPECT_NEAR(cos_pi(0.25), sqrt2_2, 1e-10);
    EXPECT_NEAR(cos_pi(0.75), -sqrt2_2, 1e-10);
    EXPECT_NEAR(cos_pi(-0.25), sqrt2_2, 1e-10);
}

TEST(CosPi, Float) {
    EXPECT_NEAR(cos_pi(0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(cos_pi(0.5f), 0.0f, 1e-6f);
    EXPECT_NEAR(cos_pi(1.0f), -1.0f, 1e-6f);
}

TEST(CosPiBackward, Integers) {
    // d/dx cos(pi*x) = -pi * sin(pi*x)
    // At x=0: -pi * sin(0) = 0
    // At x=1: -pi * sin(pi) = 0
    EXPECT_NEAR(cos_pi_backward(0.0), 0.0, 1e-10);
    EXPECT_NEAR(cos_pi_backward(1.0), 0.0, 1e-10);
    EXPECT_NEAR(cos_pi_backward(2.0), 0.0, 1e-10);
}

TEST(CosPiBackward, HalfIntegers) {
    // At x=0.5: -pi * sin(pi/2) = -pi
    // At x=1.5: -pi * sin(3*pi/2) = pi
    EXPECT_NEAR(cos_pi_backward(0.5), -M_PI, 1e-10);
    EXPECT_NEAR(cos_pi_backward(1.5), M_PI, 1e-10);
}

TEST(CosPi, ThirdValues) {
    // cos(pi/6) = sqrt(3)/2, cos(pi/3) = 1/2
    double sqrt3_2 = std::sqrt(3.0) / 2.0;
    EXPECT_NEAR(cos_pi(1.0 / 6.0), sqrt3_2, 1e-10);
    EXPECT_NEAR(cos_pi(1.0 / 3.0), 0.5, 1e-10);
    EXPECT_NEAR(cos_pi(2.0 / 3.0), -0.5, 1e-10);
    EXPECT_NEAR(cos_pi(5.0 / 6.0), -sqrt3_2, 1e-10);
}

TEST(CosPi, Symmetry) {
    // cos is an even function: cos(-x) = cos(x)
    EXPECT_NEAR(cos_pi(-0.3), cos_pi(0.3), 1e-10);
    EXPECT_NEAR(cos_pi(-0.7), cos_pi(0.7), 1e-10);
    EXPECT_NEAR(cos_pi(-1.3), cos_pi(1.3), 1e-10);
}

TEST(CosPi, NaN) {
    EXPECT_TRUE(std::isnan(cos_pi(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(cos_pi(std::numeric_limits<float>::quiet_NaN())));
}

TEST(CosPi, Infinity) {
    // cos(infinity) is undefined, should return NaN
    EXPECT_TRUE(std::isnan(cos_pi(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(cos_pi(-std::numeric_limits<double>::infinity())));
}

TEST(CosPi, LargeArgument) {
    // cos_pi should handle large arguments accurately due to periodicity
    EXPECT_NEAR(cos_pi(100.0), 1.0, 1e-10);   // 100 mod 2 = 0
    EXPECT_NEAR(cos_pi(100.5), 0.0, 1e-10);   // 100.5 mod 2 = 0.5
    EXPECT_NEAR(cos_pi(101.0), -1.0, 1e-10);  // 101 mod 2 = 1
    EXPECT_NEAR(cos_pi(-100.0), 1.0, 1e-10);
}

TEST(CosPiBackward, NumericalDerivative) {
    // Verify backward matches numerical derivative
    double x = 0.3;
    double h = 1e-7;
    double numerical = (cos_pi(x + h) - cos_pi(x - h)) / (2.0 * h);
    double analytical = cos_pi_backward(x);
    EXPECT_NEAR(analytical, numerical, 1e-5);

    x = 1.7;
    numerical = (cos_pi(x + h) - cos_pi(x - h)) / (2.0 * h);
    analytical = cos_pi_backward(x);
    EXPECT_NEAR(analytical, numerical, 1e-5);
}

TEST(CosPiBackward, Float) {
    EXPECT_NEAR(cos_pi_backward(0.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(cos_pi_backward(0.5f), -static_cast<float>(M_PI), 1e-5f);
}

TEST(CosPi, Periodicity) {
    // cos_pi has period 2: cos_pi(x + 2) = cos_pi(x)
    for (double x = -1.0; x <= 1.0; x += 0.25) {
        EXPECT_NEAR(cos_pi(x), cos_pi(x + 2.0), 1e-10);
        EXPECT_NEAR(cos_pi(x), cos_pi(x + 4.0), 1e-10);
        EXPECT_NEAR(cos_pi(x), cos_pi(x - 2.0), 1e-10);
    }
}

TEST(CosPi, Bounds) {
    // -1 <= cos_pi(x) <= 1 for all x
    for (double x = -10.0; x <= 10.0; x += 0.25) {
        EXPECT_GE(cos_pi(x), -1.0 - 1e-10);
        EXPECT_LE(cos_pi(x), 1.0 + 1e-10);
    }
}

TEST(CosPi, RelationToStdCos) {
    // cos_pi(x) = cos(pi*x)
    for (double x = -2.0; x <= 2.0; x += 0.1) {
        EXPECT_NEAR(cos_pi(x), std::cos(M_PI * x), 1e-10);
    }
}

TEST(CosPiBackward, Periodicity) {
    // derivative also has period 2: d/dx cos_pi(x + 2) = d/dx cos_pi(x)
    for (double x = -1.0; x <= 1.0; x += 0.25) {
        EXPECT_NEAR(cos_pi_backward(x), cos_pi_backward(x + 2.0), 1e-10);
    }
}

TEST(CosPiBackward, NaN) {
    EXPECT_TRUE(std::isnan(cos_pi_backward(std::numeric_limits<double>::quiet_NaN())));
}

TEST(CosPiBackward, Infinity) {
    // derivative at infinity is undefined
    EXPECT_TRUE(std::isnan(cos_pi_backward(std::numeric_limits<double>::infinity())));
}
