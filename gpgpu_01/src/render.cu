#include "render.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


// struct rgba8_t {
//   std::uint8_t r;
//   std::uint8_t g;
//   std::uint8_t b;
//   std::uint8_t a;
// };

__device__ rgba8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else if (x < 1.0)
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
  else
  {
    return rgba8_t{0, 0, 0, 255};
  }
}

__device__ uchar4 palette(int x, int N)
{
  uint8_t v = 255 * x / N;
  return {v,v,v,255};
}


/// Compute the number or iteration of the fractal per pixel and store the result in *buffer*.
/// Note that a 32-bits location can be used to store an integer (int32) or a color (uchar4). 
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void compute_iter(char* buffer, int width, int height, size_t pitch, int max_iter) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;


  uint32_t* color = (uint32_t*)(buffer + y * pitch);

  float XMIN = -2.5;
  float XMAX = 1.0;
  float YMIN = -1.0;
  float YMAX = 1.0;

  // mx0 = scaled px coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.5, 1))
  float mx0 = (x * (XMAX - XMIN) / width + XMIN);
  // my0 = scaled py coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))
  float my0 = (y * (YMIN - YMAX) / height + YMAX);

  float mx = 0.0;
  float my = 0.0;

  uint32_t i = 0;

  while (mx*mx + my*my < 2*2  && i < max_iter) {
    float mxtemp = mx*mx - my*my + mx0;
    my = 2*mx*my + my0;
    mx = mxtemp;
    i++;
  }

  color[x] = i;
}

/// This function is single thread for now!
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
/// \param LUT Output look-up table 
__global__ void compute_LUT(const char* buffer, int width, int height, size_t pitch, int max_iter, uint32_t* LUT) {

  uint32_t* histo = LUT;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      
      uint32_t* color = (uint32_t*)(buffer + y * pitch);
      uint32_t k = color[x];
      histo[k]++;
    }
  }

  rgba8_t* LUT_rgb = (rgba8_t*)LUT;

  for (int i = 1; i <= max_iter; i++) {
    histo[i] += histo[i - 1];
  }

  for (int k = 0; k < max_iter; k++) {
    float div = static_cast<float>(histo[k]) / histo[max_iter - 1];

    LUT_rgb[k] = heat_lut(div);
  }
  
  LUT_rgb[max_iter] = heat_lut(1.0f);

}


///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void apply_LUT(char* buffer, int width, int height, size_t pitch, int max_iter, const uint32_t* LUT) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
      return;


  uint32_t* color = (uint32_t*)(buffer + y * pitch);
  color[x] = LUT[color[x]];
}

// Device code
__global__ void mandelbrot(char* buffer, int width, int height, size_t pitch, int N = 100)
{
  // float denum = width * width + height * height;

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)

    return;


  rgba8_t* color = (rgba8_t*)(buffer + y * pitch);
  // float    v       = (x * x + y * y) / denum;
  // uint8_t  grayv   = v * 255;

  // color[x] = {grayv, grayv, grayv, 255};

  float XMIN = -2.5;
  float XMAX = 1.0;
  float YMIN = -1.0;
  float YMAX = 1.0;

  // mx0 = scaled px coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.5, 1))
  float mx0 = (x * (XMAX - XMIN) / width + XMIN);
  // my0 = scaled py coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))
  float my0 = (y * (YMIN - YMAX) / height + YMAX);

  float mx = 0.0;
  float my = 0.0;

  int i = 0;

  while (mx*mx + my*my < 2*2  && i < N) {
    float mxtemp = mx*mx - my*my + mx0;
    my = 2*mx*my + my0;
    mx = mxtemp;
    i++;
  }

  // color[x] = palette(i, N);

  float normalized = (float)i / N;

  color[x] = heat_lut(normalized);
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer, *LUT;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

  rc = cudaMalloc(&LUT, (n_iterations + 1) * sizeof(rgba8_t));
  if (rc)
    abortError("Fail buffer allocation");

  cudaMemset( LUT, 0, (n_iterations + 1) * sizeof(uint32_t));
  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);
    
    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    // mandelbrot<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations);
    compute_iter<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations);
    compute_LUT<<<1, 1>>>(devBuffer, width, height, pitch, n_iterations, (uint32_t*)LUT);
    apply_LUT<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations, (uint32_t*)LUT);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}
