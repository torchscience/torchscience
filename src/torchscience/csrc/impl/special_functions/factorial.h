#pragma once

/*
 * Factorial Function n!
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    For non-negative integers n:
 *        n! = 1 * 2 * 3 * ... * n
 *        0! = 1 (by convention)
 *
 *    For real/complex z (generalized via gamma function):
 *        z! = Γ(z + 1)
 *
 * 2. SPECIAL VALUES:
 *    - 0! = 1
 *    - n! = n * (n-1)! for positive integers
 *    - (-n)! = ±∞ for positive integers n (poles of gamma function)
 *
 * 3. IMPLEMENTATION:
 *    - Uses precomputed lookup tables (LUTs) for integer arguments
 *    - Float32 LUT: 0! to 34! (35! overflows float32)
 *    - Float64 LUT: 0! to 170! (171! overflows float64)
 *    - For non-integer arguments, delegates to gamma(z + 1)
 *    - Half-precision types compute in float32 for accuracy
 *
 * 4. DERIVATIVE FORMULAS:
 *    Since z! = Γ(z + 1):
 *        d/dz z! = Γ(z + 1) * ψ(z + 1)
 *    where ψ is the digamma function.
 *
 *    Second derivative:
 *        d²/dz² z! = Γ(z + 1) * (ψ(z + 1)² + ψ'(z + 1))
 *
 * 5. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Supports complex64, complex128
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

#include "is_nonpositive_integer.h"

namespace torchscience::impl::special_functions {

// ============================================================================
// Lookup table sizes
// ============================================================================

// Maximum n for which n! fits in float32 (~3.4e38)
// 34! ≈ 2.95e38, 35! ≈ 1.03e40 (overflow)
constexpr int kFactorialMaxFloat = 34;

// Maximum n for which n! fits in float64 (~1.8e308)
// 170! ≈ 7.26e306, 171! ≈ 1.24e309 (overflow)
constexpr int kFactorialMaxDouble = 170;

// For gamma: Γ(n) = (n-1)!, so max n is one more than factorial max
constexpr int kGammaMaxIntFloat = kFactorialMaxFloat + 1;   // 35
constexpr int kGammaMaxIntDouble = kFactorialMaxDouble + 1; // 171

// ============================================================================
// Float32 lookup table (0! to 34!)
// ============================================================================

constexpr float kFactorialTableFloat[kFactorialMaxFloat + 1] = {
  1.0f,                        // 0!
  1.0f,                        // 1!
  2.0f,                        // 2!
  6.0f,                        // 3!
  24.0f,                       // 4!
  120.0f,                      // 5!
  720.0f,                      // 6!
  5040.0f,                     // 7!
  40320.0f,                    // 8!
  362880.0f,                   // 9!
  3628800.0f,                  // 10!
  39916800.0f,                 // 11!
  479001600.0f,                // 12!
  6227020800.0f,               // 13!
  87178291200.0f,              // 14!
  1307674368000.0f,            // 15!
  20922789888000.0f,           // 16!
  355687428096000.0f,          // 17!
  6402373705728000.0f,         // 18!
  121645100408832000.0f,       // 19!
  2432902008176640000.0f,      // 20!
  51090942171709440000.0f,     // 21!
  1124000727777607680000.0f,   // 22!
  25852016738884976640000.0f,  // 23!
  620448401733239439360000.0f, // 24!
  15511210043330985984000000.0f,          // 25!
  403291461126605635584000000.0f,         // 26!
  10888869450418352160768000000.0f,       // 27!
  304888344611713860501504000000.0f,      // 28!
  8841761993739701954543616000000.0f,     // 29!
  265252859812191058636308480000000.0f,   // 30!
  8222838654177922817725562880000000.0f,  // 31!
  263130836933693530167218012160000000.0f,           // 32!
  8683317618811886495518194401280000000.0f,          // 33!
  295232799039604140847618609643520000000.0f,        // 34!
};

// ============================================================================
// Float64 lookup table (0! to 170!)
// ============================================================================

constexpr double kFactorialTableDouble[kFactorialMaxDouble + 1] = {
  1.0,                                           // 0!
  1.0,                                           // 1!
  2.0,                                           // 2!
  6.0,                                           // 3!
  24.0,                                          // 4!
  120.0,                                         // 5!
  720.0,                                         // 6!
  5040.0,                                        // 7!
  40320.0,                                       // 8!
  362880.0,                                      // 9!
  3628800.0,                                     // 10!
  39916800.0,                                    // 11!
  479001600.0,                                   // 12!
  6227020800.0,                                  // 13!
  87178291200.0,                                 // 14!
  1307674368000.0,                               // 15!
  20922789888000.0,                              // 16!
  355687428096000.0,                             // 17!
  6402373705728000.0,                            // 18!
  121645100408832000.0,                          // 19!
  2432902008176640000.0,                         // 20!
  51090942171709440000.0,                        // 21!
  1124000727777607680000.0,                      // 22!
  25852016738884976640000.0,                     // 23!
  620448401733239439360000.0,                    // 24!
  15511210043330985984000000.0,                  // 25!
  403291461126605635584000000.0,                 // 26!
  10888869450418352160768000000.0,               // 27!
  304888344611713860501504000000.0,              // 28!
  8841761993739701954543616000000.0,             // 29!
  265252859812191058636308480000000.0,           // 30!
  8.2228386541779228e+33,                        // 31!
  2.6313083693369353e+35,                        // 32!
  8.6833176188118865e+36,                        // 33!
  2.9523279903960414e+38,                        // 34!
  1.0333147966386145e+40,                        // 35!
  3.7199332678990122e+41,                        // 36!
  1.3763753091226346e+43,                        // 37!
  5.2302261746660111e+44,                        // 38!
  2.0397882081197444e+46,                        // 39!
  8.1591528324789773e+47,                        // 40!
  3.3452526613163807e+49,                        // 41!
  1.4050061177528799e+51,                        // 42!
  6.0415263063373836e+52,                        // 43!
  2.6582715747884488e+54,                        // 44!
  1.1962222086548019e+56,                        // 45!
  5.5026221598120889e+57,                        // 46!
  2.5862324151116818e+59,                        // 47!
  1.2413915592536073e+61,                        // 48!
  6.0828186403426756e+62,                        // 49!
  3.0414093201713378e+64,                        // 50!
  1.5511187532873822e+66,                        // 51!
  8.0658175170943878e+67,                        // 52!
  4.2748832840600255e+69,                        // 53!
  2.3084369733924138e+71,                        // 54!
  1.2696403353658276e+73,                        // 55!
  7.1099858780486345e+74,                        // 56!
  4.0526919504877214e+76,                        // 57!
  2.3505613312828785e+78,                        // 58!
  1.3868311854568984e+80,                        // 59!
  8.3209871127413901e+81,                        // 60!
  5.0758021387722480e+83,                        // 61!
  3.1469973260387937e+85,                        // 62!
  1.9826083154044401e+87,                        // 63!
  1.2688693218588417e+89,                        // 64!
  8.2476505920824707e+90,                        // 65!
  5.4434493907744307e+92,                        // 66!
  3.6471110918188685e+94,                        // 67!
  2.4800355424368305e+96,                        // 68!
  1.7112245242814131e+98,                        // 69!
  1.1978571669969892e+100,                       // 70!
  8.5047858856786232e+101,                       // 71!
  6.1234458376886087e+103,                       // 72!
  4.4701154615126844e+105,                       // 73!
  3.3078854415193864e+107,                       // 74!
  2.4809140811395398e+109,                       // 75!
  1.8854947016660503e+111,                       // 76!
  1.4518309202828587e+113,                       // 77!
  1.1324281178206297e+115,                       // 78!
  8.9461821307829753e+116,                       // 79!
  7.1569457046263802e+118,                       // 80!
  5.7971260207473680e+120,                       // 81!
  4.7536433370128418e+122,                       // 82!
  3.9455239697206587e+124,                       // 83!
  3.3142401345653533e+126,                       // 84!
  2.8171041143805503e+128,                       // 85!
  2.4227095383672732e+130,                       // 86!
  2.1077572983795277e+132,                       // 87!
  1.8548264225739844e+134,                       // 88!
  1.6507955160908461e+136,                       // 89!
  1.4857159644817615e+138,                       // 90!
  1.3520015276784030e+140,                       // 91!
  1.2438414054641307e+142,                       // 92!
  1.1567725070816416e+144,                       // 93!
  1.0873661566567430e+146,                       // 94!
  1.0329978488239059e+148,                       // 95!
  9.9167793487094969e+149,                       // 96!
  9.6192759682482120e+151,                       // 97!
  9.4268904488832478e+153,                       // 98!
  9.3326215443944153e+155,                       // 99!
  9.3326215443944153e+157,                       // 100!
  9.4259477598383594e+159,                       // 101!
  9.6144667150351266e+161,                       // 102!
  9.9029007164861805e+163,                       // 103!
  1.0299016745145628e+166,                       // 104!
  1.0813967582402909e+168,                       // 105!
  1.1462805637347084e+170,                       // 106!
  1.2265202031961380e+172,                       // 107!
  1.3246418194518290e+174,                       // 108!
  1.4438595832024936e+176,                       // 109!
  1.5882455415227430e+178,                       // 110!
  1.7629525510902446e+180,                       // 111!
  1.9745068572210740e+182,                       // 112!
  2.2311927486598138e+184,                       // 113!
  2.5435597334721876e+186,                       // 114!
  2.9250936934930159e+188,                       // 115!
  3.3931086844518982e+190,                       // 116!
  3.9699371608087209e+192,                       // 117!
  4.6845258497542907e+194,                       // 118!
  5.5745857612076058e+196,                       // 119!
  6.6895029134491271e+198,                       // 120!
  8.0942985252734437e+200,                       // 121!
  9.8750442008336013e+202,                       // 122!
  1.2146304367025330e+205,                       // 123!
  1.5061417415111409e+207,                       // 124!
  1.8826771768889261e+209,                       // 125!
  2.3721732428800469e+211,                       // 126!
  3.0126600184576594e+213,                       // 127!
  3.8562048236258040e+215,                       // 128!
  4.9745042224772874e+217,                       // 129!
  6.4668554892204737e+219,                       // 130!
  8.4715806908788205e+221,                       // 131!
  1.1182486511960043e+224,                       // 132!
  1.4872707060906857e+226,                       // 133!
  1.9929427461615188e+228,                       // 134!
  2.6904727073180504e+230,                       // 135!
  3.6590428819525487e+232,                       // 136!
  5.0128887482749920e+234,                       // 137!
  6.9177864726194885e+236,                       // 138!
  9.6157231969410890e+238,                       // 139!
  1.3462012475717526e+241,                       // 140!
  1.8981437590761709e+243,                       // 141!
  2.6953641378881628e+245,                       // 142!
  3.8543707171800728e+247,                       // 143!
  5.5502938327393044e+249,                       // 144!
  8.0479260574719919e+251,                       // 145!
  1.1749972043909107e+254,                       // 146!
  1.7272458904546389e+256,                       // 147!
  2.5563239178728654e+258,                       // 148!
  3.8089226376305697e+260,                       // 149!
  5.7133839564458546e+262,                       // 150!
  8.6272097742332404e+264,                       // 151!
  1.3113358856834524e+267,                       // 152!
  2.0063439050956823e+269,                       // 153!
  3.0897696138473508e+271,                       // 154!
  4.7891429014633939e+273,                       // 155!
  7.4710629262828944e+275,                       // 156!
  1.1729568794264145e+278,                       // 157!
  1.8532718694937346e+280,                       // 158!
  2.9467022724950384e+282,                       // 159!
  4.7147236359920616e+284,                       // 160!
  7.5907050539472187e+286,                       // 161!
  1.2296942187394494e+289,                       // 162!
  2.0044015765453026e+291,                       // 163!
  3.2872185855342959e+293,                       // 164!
  5.4239106661315888e+295,                       // 165!
  9.0036917057784373e+297,                       // 166!
  1.5036165148649989e+300,                       // 167!
  2.5260757449731984e+302,                       // 168!
  4.2690680090047051e+304,                       // 169!
  7.2574156153079990e+306,                       // 170!
};

// ============================================================================
// Forward declarations for gamma function
// ============================================================================

// Forward declare gamma function templates (defined in gamma.h)
// These are needed for non-integer factorial computation: z! = Γ(z + 1)

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
gamma(scalar_t z);

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  !std::is_same_v<scalar_t, float> &&
  !std::is_same_v<scalar_t, double>,
  scalar_t>
gamma(scalar_t z);

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<c10::is_complex<scalar_t>::value, scalar_t>
gamma(scalar_t z);

// ============================================================================
// Helper: check if value is a non-negative integer
// ============================================================================

/**
 * Check if a floating-point value is a non-negative integer.
 * Returns true if z is exactly 0, 1, 2, 3, ...
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<scalar_t>::value, bool>
is_nonnegative_integer(scalar_t z) {
  using std::floor;
  return z >= scalar_t(0) && z == floor(z);
}

/**
 * Check if a complex value is a non-negative integer (real, with zero imaginary part).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool
is_nonnegative_integer(c10::complex<T> z) {
  using std::abs;
  using std::floor;

  // Type-appropriate tolerance
  T tol;
  if constexpr (std::is_same_v<T, double>) {
    tol = T(kPoleDetectionToleranceDouble);
  } else {
    tol = T(kPoleDetectionToleranceFloat);
  }

  // Check imaginary part is approximately zero
  if (abs(z.imag()) > tol) {
    return false;
  }

  T real_part = z.real();

  // Check if real part is non-negative and approximately an integer
  if (real_part < -tol) {
    return false;
  }

  T rounded = floor(real_part + T(0.5));
  return abs(real_part - rounded) <= tol * (T(1) + abs(rounded));
}

// ============================================================================
// Factorial function forward implementation
// ============================================================================

/**
 * Factorial function for float using LUT.
 *
 * For non-negative integers 0 <= n <= 34, returns n! from lookup table.
 * For n > 34, returns infinity (overflow).
 * For negative integers, returns infinity (poles of gamma function).
 * For non-integer values, computes via gamma(z + 1).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<std::is_same_v<scalar_t, float>, scalar_t>
factorial(scalar_t z) {
  using std::floor;

  // Handle NaN
  if (z != z) {
    return z;
  }

  // Check if z is a non-negative integer
  if (is_nonnegative_integer(z)) {
    int n = static_cast<int>(z);
    if (n <= kFactorialMaxFloat) {
      return kFactorialTableFloat[n];
    }
    // n > 34: overflow
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Check for negative integers (poles)
  if (z < scalar_t(0) && z == floor(z)) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Non-integer: use gamma(z + 1)
  return gamma(z + scalar_t(1));
}

/**
 * Factorial function for double using LUT.
 *
 * For non-negative integers 0 <= n <= 170, returns n! from lookup table.
 * For n > 170, returns infinity (overflow).
 * For negative integers, returns infinity (poles of gamma function).
 * For non-integer values, computes via gamma(z + 1).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<std::is_same_v<scalar_t, double>, scalar_t>
factorial(scalar_t z) {
  using std::floor;

  // Handle NaN
  if (z != z) {
    return z;
  }

  // Check if z is a non-negative integer
  if (is_nonnegative_integer(z)) {
    int n = static_cast<int>(z);
    if (n <= kFactorialMaxDouble) {
      return kFactorialTableDouble[n];
    }
    // n > 170: overflow
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Check for negative integers (poles)
  if (z < scalar_t(0) && z == floor(z)) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  // Non-integer: use gamma(z + 1)
  return gamma(z + scalar_t(1));
}

/**
 * Factorial function for half-precision types.
 * Computes in float32 for accuracy, then converts back.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  !std::is_same_v<scalar_t, float> &&
  !std::is_same_v<scalar_t, double>,
  scalar_t>
factorial(scalar_t z) {
  return static_cast<scalar_t>(factorial(static_cast<float>(z)));
}

/**
 * Factorial function for complex types.
 *
 * For complex numbers that are non-negative integers (real with zero imaginary),
 * uses the lookup table. Otherwise computes via gamma(z + 1).
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<c10::is_complex<scalar_t>::value, scalar_t>
factorial(scalar_t z) {
  using T = typename scalar_t::value_type;
  using std::abs;
  using std::floor;
  using std::isnan;

  // Handle NaN propagation
  if (isnan(z.real()) || isnan(z.imag())) {
    return scalar_t(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  // Type-appropriate tolerance
  T tol;
  if constexpr (std::is_same_v<T, double>) {
    tol = T(kPoleDetectionToleranceDouble);
  } else {
    tol = T(kPoleDetectionToleranceFloat);
  }

  // For values on or very close to the real axis with non-negative integer real part,
  // use the lookup table
  if (abs(z.imag()) <= tol) {
    T real_part = z.real();

    // Check if approximately a non-negative integer
    if (real_part >= -tol) {
      T rounded = floor(real_part + T(0.5));
      if (abs(real_part - rounded) <= tol * (T(1) + abs(rounded))) {
        int n = static_cast<int>(rounded);
        if constexpr (std::is_same_v<T, double>) {
          if (n <= kFactorialMaxDouble) {
            return scalar_t(kFactorialTableDouble[n], T(0));
          }
        } else {
          if (n <= kFactorialMaxFloat) {
            return scalar_t(kFactorialTableFloat[n], T(0));
          }
        }
        // Overflow
        return scalar_t(std::numeric_limits<T>::infinity(), T(0));
      }
    }

    // Check for negative integer (pole)
    if (real_part < -tol) {
      T rounded = floor(real_part + T(0.5));
      if (abs(real_part - rounded) <= tol * (T(1) + abs(rounded))) {
        return scalar_t(std::numeric_limits<T>::infinity(), T(0));
      }
    }
  }

  // General case: z! = Γ(z + 1)
  return gamma(z + scalar_t(T(1), T(0)));
}

}  // namespace torchscience::impl::special_functions
