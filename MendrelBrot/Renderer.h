#pragma once
#include <memory>
#include "ComputeMethods.h"
#include "olcPixelGameEngine.h"

class Renderer
{
public:
	void RenderFractal(ComputeMethods& method , int screenwidth, int screenheight , olc::PixelGameEngine* engine);
	void RenderUI(olc::PixelGameEngine* engine, std::chrono::duration<double>& dt , const std::string& simulation_type, int maxitertaions);
};