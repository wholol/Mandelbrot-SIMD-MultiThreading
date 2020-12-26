#include "Renderer.h"


void Renderer::RenderFractal(ComputeMethods& method, int screenwidth, int screenheight , olc::PixelGameEngine* engine)
{
	for (int y = 0; y < screenheight; ++y)
	{
		for (int x = 0; x < screenwidth; ++x)
		{
			int& n = method.get_iteration_vec()[x + screenwidth * y];
			static constexpr float a = 0.1f;
			engine->Draw(x, y, olc::PixelF(0.5f * sinf(a * n) + 0.5f, 0.5f * sinf(a * n + 2.094f) + 0.5f, 0.5f * sinf(a * n + 4.188f) + 0.5f));
		}
	}
}



void Renderer::RenderUI(olc::PixelGameEngine* engine, std::chrono::duration<double>& dt, const std::string& simulation_type,int maxiterations,int timesamples, int maxsamples)
{
	if (timesamples < maxsamples)
	{
		engine->DrawString(0, 90, "Sampling data for plotting..: " + std::to_string(timesamples), olc::BLACK, 3);
	}
	else {
		engine->DrawString(0, 90, "Sampling complete!" , olc::BLACK, 3);
	}
	engine->DrawString(0, 0, "Simulation Type: " + simulation_type , olc::BLACK, 3);
	engine->DrawString(0, 30, "Time Taken: " + std::to_string(dt.count()) + "s", olc::BLACK, 3);
	engine->DrawString(0, 60, "Iterations: " + std::to_string(maxiterations), olc::BLACK, 3);
	
}
