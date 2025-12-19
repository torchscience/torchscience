#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "impl/special_functions/polygamma.h"

using namespace torchscience::impl::special_functions;

class PolygammaTest : public ::testing::Test {
 protected:
  static constexpr double kPi = 3.14159265358979323846;
};

// ============================================================================
// Dispatch to Specialized Functions Tests
// ============================================================================

TEST_F(PolygammaTest, Double_DispatchToTrigamma) {
  // polygamma(1, x) should equal trigamma(x)
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0, 10.0};
  for (double x : test_values) {
    double poly_result = polygamma(1, x);
    double tri_result = trigamma(x);
    EXPECT_NEAR(poly_result, tri_result, std::abs(tri_result) * 1e-10)
        << "Failed for x = " << x;
  }
}

TEST_F(PolygammaTest, Double_DispatchToTetragamma) {
  // polygamma(2, x) should equal tetragamma(x)
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0, 10.0};
  for (double x : test_values) {
    double poly_result = polygamma(2, x);
    double tetra_result = tetragamma(x);
    EXPECT_NEAR(poly_result, tetra_result, std::abs(tetra_result) * 1e-10)
        << "Failed for x = " << x;
  }
}

TEST_F(PolygammaTest, Double_DispatchToPentagamma) {
  // polygamma(3, x) should equal pentagamma(x)
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0, 10.0};
  for (double x : test_values) {
    double poly_result = polygamma(3, x);
    double penta_result = pentagamma(x);
    EXPECT_NEAR(poly_result, penta_result, std::abs(penta_result) * 1e-10)
        << "Failed for x = " << x;
  }
}

// ============================================================================
// Float Tests
// ============================================================================

TEST_F(PolygammaTest, Float_Order1) {
  float result = polygamma(1, 1.0f);
  float expected = static_cast<float>(kPi * kPi / 6.0);
  EXPECT_NEAR(result, expected, 1e-5f);
}

TEST_F(PolygammaTest, Float_Order2) {
  float result = polygamma(2, 1.0f);
  float expected = tetragamma(1.0f);
  EXPECT_NEAR(result, expected, 1e-5f);
}

TEST_F(PolygammaTest, Float_Order3) {
  float result = polygamma(3, 1.0f);
  float expected = pentagamma(1.0f);
  EXPECT_NEAR(result, expected, 1e-4f);
}

TEST_F(PolygammaTest, Float_Poles) {
  EXPECT_TRUE(std::isnan(polygamma(1, 0.0f)));
  EXPECT_TRUE(std::isnan(polygamma(2, 0.0f)));
  EXPECT_TRUE(std::isnan(polygamma(3, 0.0f)));
  EXPECT_TRUE(std::isnan(polygamma(1, -1.0f)));
}

TEST_F(PolygammaTest, Float_InvalidOrder) {
  EXPECT_TRUE(std::isnan(polygamma(-1, 1.0f)));
}

TEST_F(PolygammaTest, Float_NaN) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(polygamma(1, nan_val)));
}

// ============================================================================
// Double Tests
// ============================================================================

TEST_F(PolygammaTest, Double_Order1) {
  double result = polygamma(1, 1.0);
  double expected = kPi * kPi / 6.0;
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(PolygammaTest, Double_Order2) {
  double result = polygamma(2, 1.0);
  double expected = tetragamma(1.0);
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(PolygammaTest, Double_Order3) {
  double result = polygamma(3, 1.0);
  double expected = pentagamma(1.0);
  EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(PolygammaTest, Double_Poles) {
  EXPECT_TRUE(std::isnan(polygamma(1, 0.0)));
  EXPECT_TRUE(std::isnan(polygamma(2, 0.0)));
  EXPECT_TRUE(std::isnan(polygamma(3, 0.0)));
  EXPECT_TRUE(std::isnan(polygamma(1, -1.0)));
  EXPECT_TRUE(std::isnan(polygamma(2, -2.0)));
}

TEST_F(PolygammaTest, Double_InvalidOrder) {
  EXPECT_TRUE(std::isnan(polygamma(-1, 1.0)));
  EXPECT_TRUE(std::isnan(polygamma(-5, 2.0)));
}

TEST_F(PolygammaTest, Double_NaN) {
  double nan_val = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(polygamma(1, nan_val)));
  EXPECT_TRUE(std::isnan(polygamma(2, nan_val)));
}

// ============================================================================
// Higher Order Tests (n > 3)
// ============================================================================

TEST_F(PolygammaTest, Double_Order4) {
  // Test general formula for n=4
  double x = 2.0;
  double result = polygamma(4, x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

TEST_F(PolygammaTest, Double_Order5) {
  double x = 2.0;
  double result = polygamma(5, x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
}

TEST_F(PolygammaTest, Double_HigherOrderLargeX) {
  // Higher order polygamma for large x
  double x = 50.0;
  double result = polygamma(4, x);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_FALSE(std::isinf(result));
  // For large x, psi^(n)(x) should be small
  EXPECT_LT(std::abs(result), 1e-4);
}

// ============================================================================
// Recurrence Relation Tests
// ============================================================================

TEST_F(PolygammaTest, Double_RecurrenceOrder1) {
  // psi^(1)(x+1) = psi^(1)(x) - 1/x^2
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0};
  for (double x : test_values) {
    double lhs = polygamma(1, x + 1.0);
    double rhs = polygamma(1, x) - 1.0 / (x * x);
    EXPECT_NEAR(lhs, rhs, 1e-10) << "Failed for x = " << x;
  }
}

TEST_F(PolygammaTest, Double_RecurrenceOrder2) {
  // psi^(2)(x+1) = psi^(2)(x) + 2/x^3
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0};
  for (double x : test_values) {
    double lhs = polygamma(2, x + 1.0);
    double rhs = polygamma(2, x) + 2.0 / (x * x * x);
    EXPECT_NEAR(lhs, rhs, 1e-10) << "Failed for x = " << x;
  }
}

TEST_F(PolygammaTest, Double_RecurrenceOrder3) {
  // psi^(3)(x+1) = psi^(3)(x) - 6/x^4
  std::vector<double> test_values = {0.5, 1.0, 1.5, 2.0, 5.0};
  for (double x : test_values) {
    double lhs = polygamma(3, x + 1.0);
    double rhs = polygamma(3, x) - 6.0 / (x * x * x * x);
    EXPECT_NEAR(lhs, rhs, std::abs(rhs) * 1e-11) << "Failed for x = " << x;
  }
}

// ============================================================================
// Sign Pattern Tests
// ============================================================================

TEST_F(PolygammaTest, Double_SignPattern) {
  // For positive x:
  // psi^(n)(x) > 0 for odd n
  // psi^(n)(x) < 0 for even n
  double x = 2.0;

  // n=1 (trigamma) should be positive
  EXPECT_GT(polygamma(1, x), 0.0);

  // n=2 (tetragamma) should be negative
  EXPECT_LT(polygamma(2, x), 0.0);

  // n=3 (pentagamma) should be positive
  EXPECT_GT(polygamma(3, x), 0.0);
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(PolygammaTest, FloatDoubleConsistency) {
  std::vector<double> test_values = {0.5, 1.0, 2.0, 5.0};
  for (double x : test_values) {
    for (int n = 1; n <= 3; ++n) {
      float float_result = polygamma(n, static_cast<float>(x));
      double double_result = polygamma(n, x);
      EXPECT_NEAR(float_result, double_result, std::abs(double_result) * 1e-5)
          << "Failed for n = " << n << ", x = " << x;
    }
  }
}

// ============================================================================
// Negative x Tests
// ============================================================================

TEST_F(PolygammaTest, Double_NegativeNonInteger) {
  double x = -0.5;
  double result1 = polygamma(1, x);
  double result2 = polygamma(2, x);
  double result3 = polygamma(3, x);

  EXPECT_FALSE(std::isnan(result1));
  EXPECT_FALSE(std::isnan(result2));
  EXPECT_FALSE(std::isnan(result3));
  EXPECT_FALSE(std::isinf(result1));
  EXPECT_FALSE(std::isinf(result2));
  EXPECT_FALSE(std::isinf(result3));
}
