#include <gtest/gtest.h>

#include <torchscience/csrc/impl/special_functions/sin_pi.h>

#include <cmath>
#include <limits>

using torchscience::impl::special_functions::sin_pi;
using torchscience::impl::special_functions::sin_pi_backward;

TEST(SinPi, Integers) {
    // sin(n * pi) = 0 for all integers n
    EXPECT_NEAR(sin_pi(0.0), 0.0, 1e-10);
    EXPECT_NEAR(sin_pi(1.0), 0.0, 1e-10);
    EXPECT_NEAR(sin_pi(2.0), 0.0, 1e-10);
    EXPECT_NEAR(sin_pi(-1.0), 0.0, 1e-10);
    EXPECT_NEAR(sin_pi(-2.0), 0.0, 1e-10);
}

TEST(SinPi, HalfIntegers) {
    // sin(pi/2) = 1, sin(3*pi/2) = -1
    EXPECT_NEAR(sin_pi(0.5), 1.0, 1e-10);
    EXPECT_NEAR(sin_pi(1.5), -1.0, 1e-10);
    EXPECT_NEAR(sin_pi(-0.5), -1.0, 1e-10);
    EXPECT_NEAR(sin_pi(2.5), 1.0, 1e-10);
}

TEST(SinPi, QuarterValues) {
    // sin(pi/4) = sqrt(2)/2
    double sqrt2_2 = std::sqrt(2.0) / 2.0;
    EXPECT_NEAR(sin_pi(0.25), sqrt2_2, 1e-10);
    EXPECT_NEAR(sin_pi(0.75), sqrt2_2, 1e-10);
    EXPECT_NEAR(sin_pi(-0.25), -sqrt2_2, 1e-10);
}

TEST(SinPi, Float) {
    EXPECT_NEAR(sin_pi(0.5f), 1.0f, 1e-6f);
    EXPECT_NEAR(sin_pi(1.0f), 0.0f, 1e-6f);
}

TEST(SinPiBackward, Integers) {
    // d/dx sin(pi*x) = pi * cos(pi*x)
    // At x=0: pi * cos(0) = pi
    // At x=1: pi * cos(pi) = -pi
    EXPECT_NEAR(sin_pi_backward(0.0), M_PI, 1e-10);
    EXPECT_NEAR(sin_pi_backward(1.0), -M_PI, 1e-10);
    EXPECT_NEAR(sin_pi_backward(2.0), M_PI, 1e-10);
}

TEST(SinPiBackward, HalfIntegers) {
    // At x=0.5: pi * cos(pi/2) = 0
    // At x=1.5: pi * cos(3*pi/2) = 0
    EXPECT_NEAR(sin_pi_backward(0.5), 0.0, 1e-10);
    EXPECT_NEAR(sin_pi_backward(1.5), 0.0, 1e-10);
}

TEST(SinPi, ThirdValues) {
    // sin(pi/6) = 1/2, sin(pi/3) = sqrt(3)/2
    double sqrt3_2 = std::sqrt(3.0) / 2.0;
    EXPECT_NEAR(sin_pi(1.0 / 6.0), 0.5, 1e-10);
    EXPECT_NEAR(sin_pi(1.0 / 3.0), sqrt3_2, 1e-10);
    EXPECT_NEAR(sin_pi(2.0 / 3.0), sqrt3_2, 1e-10);
    EXPECT_NEAR(sin_pi(5.0 / 6.0), 0.5, 1e-10);
}

TEST(SinPi, Antisymmetry) {
    // sin is an odd function: sin(-x) = -sin(x)
    EXPECT_NEAR(sin_pi(-0.3), -sin_pi(0.3), 1e-10);
    EXPECT_NEAR(sin_pi(-0.7), -sin_pi(0.7), 1e-10);
    EXPECT_NEAR(sin_pi(-1.3), -sin_pi(1.3), 1e-10);
}

TEST(SinPi, NaN) {
    EXPECT_TRUE(std::isnan(sin_pi(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(sin_pi(std::numeric_limits<float>::quiet_NaN())));
}

TEST(SinPi, Infinity) {
    // sin(infinity) is undefined, should return NaN
    EXPECT_TRUE(std::isnan(sin_pi(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(sin_pi(-std::numeric_limits<double>::infinity())));
}

TEST(SinPi, LargeArgument) {
    // sin_pi should handle large arguments accurately due to periodicity
    // sin_pi(x) = sin_pi(x mod 2) for the effective period
    EXPECT_NEAR(sin_pi(100.5), 1.0, 1e-10);  // 100.5 mod 2 = 0.5
    EXPECT_NEAR(sin_pi(101.0), 0.0, 1e-10);  // 101 mod 2 = 1
    EXPECT_NEAR(sin_pi(-100.5), -1.0, 1e-10);
}

TEST(SinPiBackward, NumericalDerivative) {
    // Verify backward matches numerical derivative
    double x = 0.3;
    double h = 1e-7;
    double numerical = (sin_pi(x + h) - sin_pi(x - h)) / (2.0 * h);
    double analytical = sin_pi_backward(x);
    EXPECT_NEAR(analytical, numerical, 1e-5);

    x = 1.7;
    numerical = (sin_pi(x + h) - sin_pi(x - h)) / (2.0 * h);
    analytical = sin_pi_backward(x);
    EXPECT_NEAR(analytical, numerical, 1e-5);
}

TEST(SinPiBackward, Float) {
    EXPECT_NEAR(sin_pi_backward(0.0f), static_cast<float>(M_PI), 1e-5f);
    EXPECT_NEAR(sin_pi_backward(0.5f), 0.0f, 1e-5f);
}

TEST(SinPi, Periodicity) {
    // sin_pi has period 2: sin_pi(x + 2) = sin_pi(x)
    for (double x = -1.0; x <= 1.0; x += 0.25) {
        EXPECT_NEAR(sin_pi(x), sin_pi(x + 2.0), 1e-10);
        EXPECT_NEAR(sin_pi(x), sin_pi(x + 4.0), 1e-10);
        EXPECT_NEAR(sin_pi(x), sin_pi(x - 2.0), 1e-10);
    }
}

TEST(SinPi, Bounds) {
    // -1 <= sin_pi(x) <= 1 for all x
    for (double x = -10.0; x <= 10.0; x += 0.25) {
        EXPECT_GE(sin_pi(x), -1.0 - 1e-10);
        EXPECT_LE(sin_pi(x), 1.0 + 1e-10);
    }
}

TEST(SinPi, RelationToStdSin) {
    // sin_pi(x) = sin(pi*x)
    for (double x = -2.0; x <= 2.0; x += 0.1) {
        EXPECT_NEAR(sin_pi(x), std::sin(M_PI * x), 1e-10);
    }
}

TEST(SinPiBackward, Periodicity) {
    // derivative also has period 2: d/dx sin_pi(x + 2) = d/dx sin_pi(x)
    for (double x = -1.0; x <= 1.0; x += 0.25) {
        EXPECT_NEAR(sin_pi_backward(x), sin_pi_backward(x + 2.0), 1e-10);
    }
}

TEST(SinPiBackward, NaN) {
    EXPECT_TRUE(std::isnan(sin_pi_backward(std::numeric_limits<double>::quiet_NaN())));
}

TEST(SinPiBackward, Infinity) {
    // derivative at infinity is undefined
    EXPECT_TRUE(std::isnan(sin_pi_backward(std::numeric_limits<double>::infinity())));
}
