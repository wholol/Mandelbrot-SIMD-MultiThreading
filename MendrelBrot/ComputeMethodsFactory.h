#pragma once
#include <memory>
#include "ComputeMethods.h"
#include "olcPixelGameEngine.h"

class ComputeMethodsFactory
{

public:
	static std::unique_ptr<ComputeMethods> create( int screenheight , int screenwidth, int num_threads, int simulation_type)
	{
		if (simulation_type == 1)
		{
			return std::make_unique<frac_basic>(screenwidth, screenheight, num_threads);
		}

		else if (simulation_type == 2)
		{
			return std::make_unique<frac_basic_SIMD>(screenwidth, screenheight, num_threads);
		}

		else if (simulation_type == 3)
		{
			return std::make_unique<frac_multithread_SIMD>(screenwidth, screenheight, num_threads);
		}

		else if (simulation_type == 4)
		{
			return std::make_unique<frac_threadpool_SIMD>(screenwidth, screenheight, num_threads);
		}

		else if (simulation_type == 5)
		{
			return std::make_unique<frac_async_SIMD>(screenwidth, screenheight, num_threads);
		}

			return nullptr;
	}

};