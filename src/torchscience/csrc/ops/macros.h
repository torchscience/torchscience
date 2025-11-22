#pragma once

#if defined(_WIN32) && !defined(TORCHSCIENCE_BUILD_STATIC_LIBS)
#if defined(torchscience_EXPORTS)
#define SCIENCE_API __declspec(dllexport)
#else
#define SCIENCE_API __declspec(dllimport)
#endif
#else
#define SCIENCE_API
#endif
