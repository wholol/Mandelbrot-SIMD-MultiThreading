#pragma once
#include <vector>

#include "olcPixelGameEngine.h"
#include "ThreadPool.h"

class ComputeMethods
{
public:
	ComputeMethods(int thread_num);
	void set_screen_params(int screenwidth, int screenheight);		//MUST CALL THIS FUNCTION TO WORK
	void frac_basic(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations);
	void frac_SIMD(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations);
	void frac_multithread(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations);
	void frac_threadpool(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations);
	void frac_async(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations);
	std::vector<int>& get_iteration_vec();

private:
	int screenwidth, screenheight, thread_num;
	std::vector<int> iteration_vec;
	ThreadPool pool;
};


