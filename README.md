# Mandelbrot-SIMD-Multithreading
MandelBrot program to understand and compare the performance of SIMD and threading, threadpools and std::async. olcPixelGameEngine by javidx9 is used.

this program compares the performance of computing the Mandrelbrot set with various optimizations. The methods used are:

default algorithm  
SIMD  
multithreading and SIMD  
threadpool and SIMD  
std::async and SIMD  

<img src="MendrelBrot/SIMD.png"> 

## Benchmark Results
CPU : intel i7-7700 HQ (4 cores 8 threads)  
This benchmark is roughly the whole Mandrelbot set. In reality, the set can be zoomed in and out which will cause different results, dpeending on the region. Overall, the threadpool and std::async was the fastest, with minimal differences.  
  
<img src="MendrelBrot/benchmark.png"> 
