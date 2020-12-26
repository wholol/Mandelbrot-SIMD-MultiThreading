#pragma once
#include <memory>
#include "ComputeMethods.h"
#include "olcPixelGameEngine.h"

class Renderer
{
public:
	static void RenderFractal(ComputeMethods& method , int screenwidth, int screenheight , olc::PixelGameEngine* engine);
	static void RenderUI(olc::PixelGameEngine* engine, std::chrono::duration<double>& dt , const std::string& simulation_type, int maxitertaions, int timesamples, int maxsamples);
};