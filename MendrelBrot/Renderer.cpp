#include "Renderer.h"


std::unique_ptr<ComputeMethods> Renderer::RenderFractal(std::unique_ptr<ComputeMethods> method , int screenwidth, int screenheight , olc::PixelGameEngine& engine)
{
	
	
	for (int y = 0; y < screenheight; ++y)
	{
		for (int x = 0; x < screenwidth; ++x)
		{
			int& n = method->get_iteration_vec()[x + screenwidth * y];
			static constexpr float a = 0.1f;
			engine.Draw(x, y, olc::PixelF(0.5f * sinf(a * n) + 0.5f, 0.5f * sinf(a * n + 2.094f) + 0.5f, 0.5f * sinf(a * n + 4.188f) + 0.5f));
		}
	}
	return method;
}



void Renderer::RenderUI(olc::PixelGameEngine& engine, std::chrono::duration<double>& dt, const std::string& simulation_type,int maxiterations)
{
	
	engine.DrawString(0, 0, "Simulation Type: " + simulation_type , olc::BLACK, 3);
	engine.DrawString(0, 30, "Time Taken: " + std::to_string(dt.count()) + "s", olc::BLACK, 3);
	engine.DrawString(0, 60, "Iterations: " + std::to_string(maxiterations), olc::BLACK, 3);
	
}
