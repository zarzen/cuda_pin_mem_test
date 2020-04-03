#include "sock_cli_serv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

using namespace trans;

__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

double time_now() {
  auto t = std::chrono::high_resolution_clock::now();
  return t.time_since_epoch().count() / 1e9; // convert to seconds
};

void sock_server() {
  SockServ serv("8888");
  serv._listen();
  char msg[4];
  serv._recv(msg, 4);
  serv._send("SYNC", 4);
}

void sock_cli() {
  SockCli cli("127.0.0.1", "8888");
  // std::string msg = "1234";
  cli._send("1234", 4);
  char r[4];
  cli._recv(r, 4);
}

void computation() {
  int N = 120 * (1 << 20);
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // sync
  sock_cli();
  cudaEventRecord(start);
  double s = time_now();
  // Perform SAXPY on 1M elements
  saxpy<<<(N + 511) / 512, 512>>>(N, 2.0f, d_x, d_y);

  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  double e = time_now();
  printf("chrono dur: %lf, start %lf, end %lf\n", e - s, s, e);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Dur: %f \n", milliseconds);
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0f));
  }

  printf("Max error: %f\n", maxError);
  printf("Effective Bandwidth (GB/s): %f\n", N * 4 * 3 / milliseconds / 1e6);
}



void pin_mem() {
  int N = 120 * (1 << 20);
  float *x, *y, *d_x, *d_y;
  size_t msize = N * sizeof(float);
  x = (float *)malloc(msize);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  sock_server();
  double s = time_now();
  cudaEventRecord(start);
  cudaHostRegister(x, msize, 0);

  double e = time_now();
  cudaEventRecord(stop);
  printf("(chron dur: %lf, start: %lf, end: %lf\n", e-s, s, e);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Pin mem cost dur: %f\n", milliseconds);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("usage: ./a <mode>");
  }
  char mode = std::atoi(argv[1]);
  if (mode == 0) {
    // serv 
    printf("server mode\n");
    
    pin_mem();
  } else {
    printf("cli mode\n");
    computation();

  }

}