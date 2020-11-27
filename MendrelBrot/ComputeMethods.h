#pragma once
#include <vector>
#include "olcPixelGameEngine.h"

class ComputeMethods
{
public:
	ComputeMethods(int screenwidth, int screenheight, int thread_num) 
		:screenwidth(screenwidth) , screenheight(screenheight) , thread_num(thread_num)
	{};
	virtual void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) = 0;

protected:
	std::vector<double> iteration_vec;
	int screenwidth, screenheight, thread_num;
};


class frac_basic : public ComputeMethods
{
public:
	frac_basic(int screenwidth, int screenheight, int thread_num) : ComputeMethods(screenwidth, screenheight, thread_num) {};

	void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) override;
};

class frac_basic_SIMD : public ComputeMethods
{
public:
	frac_basic_SIMD(int screenwidth, int screenheight, int thread_num) : ComputeMethods(screenwidth, screenheight, thread_num) {};

	void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) override;
};

class frac_multithread_SIMD : public ComputeMethods
{
	frac_multithread_SIMD (int screenwidth, int screenheight, int thread_num) : ComputeMethods(screenwidth, screenheight, thread_num) {};
	void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) override;
};

class frac_threadpool_SIMD : public ComputeMethods
{
	frac_threadpool_SIMD(int screenwidth, int screenheight, int thread_num) : ComputeMethods(screenwidth, screenheight, thread_num) {};
	void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) override;
};

class frac_async_SIMD : public ComputeMethods
{
	frac_async_SIMD(int screenwidth, int screenheight, int thread_num) : ComputeMethods(screenwidth, screenheight, thread_num) {};
	void compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations) override;
};