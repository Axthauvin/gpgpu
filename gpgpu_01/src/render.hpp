#pragma once
#include <cassert>
#include <memory>
#include <spdlog/spdlog.h>

struct rgba8_t
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
extern "C" void render_cpu(char *buffer, int width, int height,
                           std::ptrdiff_t stride, int n_iterations = 100);

/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render(char *buffer, int width, int height, std::ptrdiff_t stride,
            int n_iterations = 100);
