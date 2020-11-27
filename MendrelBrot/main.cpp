#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <immintrin.h>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <future>
#include <deque>
#include "ComputeMethods.h"
#include "Renderer.h"


class MendrelBrot : public olc::PixelGameEngine
{

public:
	MendrelBrot()
		:m(std::thread::hardware_concurrency())
	{
		sAppName = "MendrelBrot";
	};

public:

	bool OnUserCreate() override
	{	
		m.set_screen_params(ScreenWidth(), ScreenHeight());	//LEAVE THIS HERE
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
	
		olc::vd2d GetMousePos = { (double)GetMouseX() , (double)GetMouseY() };

		if (GetKey(olc::Key::A).bPressed)
		{
			MouseStartPos = GetMousePos;
		}

		//if left click is held
		if (GetKey(olc::Key::A).bHeld)
		{
			OffSet.x -= (GetMousePos.x - MouseStartPos.x) / Scale.x;
			OffSet.y -= (GetMousePos.y - MouseStartPos.y) / Scale.y;
			MouseStartPos = GetMousePos;
		}

		//screen coordiantes to world coordinates
		olc::vd2d MouseBeforeZoom;
		ScreenToWorld(GetMousePos, MouseBeforeZoom);

		/*zoom in*/
		if (GetKey(olc::Key::Q).bHeld)
		{
			Scale *= 1.2f;
		}

		/*zoom out*/
		if (GetKey(olc::Key::E).bHeld)
		{
			Scale *= 0.8f;
		}

		olc::vd2d MouseAfterZoom;
		ScreenToWorld(GetMousePos, MouseAfterZoom);

		olc::vd2d GetOffSetVector = MouseAfterZoom - MouseBeforeZoom;

		OffSet -= GetOffSetVector;

		if (GetKey(olc::Key::UP).bPressed)
		{
			maxiterations += 100;
		}
		if (GetKey(olc::Key::DOWN).bPressed)
		{
			
			maxiterations -= 100;
			if (maxiterations <= 50)
			{
				maxiterations = 50;
			}
		}

		//top left and bottom right of screenspace
		olc::vd2d ScreenTL = { 0 , 0 };
		olc::vd2d ScreenBR = { 1280, 720 };

		//top elft and bottom right of mendrelbrot space
		olc::vd2d FractalTL = { 0.0f , 0.0f };
		olc::vd2d FractalBR = { 3.0f , 2.0f };

		/*map the screen space to world space*/
		ScreenToWorld(ScreenTL, FractalTL);
		ScreenToWorld(ScreenBR, FractalBR);

		if ((GetKey(olc::Key::K1).bPressed)) {
			type = "normal algorithm";
		}

		if ((GetKey(olc::Key::K2).bPressed)) {
			type = "SIMD";
		}

		if ((GetKey(olc::Key::K3).bPressed)) {
			type = "multithreaded + SIMD";
		}

		if ((GetKey(olc::Key::K4).bPressed)) {
			type = "threadpool + SIMD";
		}

		if ((GetKey(olc::Key::K5).bPressed)) {
			type = "std::async + SIMD";
		}

			if (type == "normal algorithm")
			{
				auto start = std::chrono::high_resolution_clock::now();
				m.frac_basic(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
				auto end = std::chrono::high_resolution_clock::now();
				dt = end - start;
			}
			
			else if (type =="SIMD")
			{
				auto start = std::chrono::high_resolution_clock::now();
				m.frac_SIMD(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
				auto end = std::chrono::high_resolution_clock::now();
				dt = end - start;

			}

			else if (type == "multithreaded + SIMD")
			{
				auto start = std::chrono::high_resolution_clock::now();
				m.frac_multithread(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
				auto end = std::chrono::high_resolution_clock::now();
				dt = end - start;

			}

			else if (type == "threadpool + SIMD")
			{
				auto start = std::chrono::high_resolution_clock::now();
				m.frac_threadpool(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
				auto end = std::chrono::high_resolution_clock::now();
				dt = end - start;

			}

			else if (type == "std::async + SIMD")
			{
				auto start = std::chrono::high_resolution_clock::now();
				m.frac_async(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
				auto end = std::chrono::high_resolution_clock::now();
				dt = end - start;
			}
			/*render*/
			render.RenderFractal(m, ScreenWidth(), ScreenHeight(), this);
			render.RenderUI(this, dt,type, maxiterations);
		
		
		return true;
	}

private:
	ComputeMethods m;
	Renderer render;
	std::string type = "normal algorithm";
	olc::vd2d MouseStartPos = { 0 ,0 };
	olc::vd2d Scale = { 1280.0 * 0.5 , 720.0 };
	olc::vd2d OffSet = { 0.0 , 0.0 };

	
	
	std::chrono::duration<double> dt;


	int maxiterations = 50;
	 //stores the iteration counter for the fractal (purpose for colouring the fractal)
		
	 void WorldToScreen(const olc::vd2d& v, olc::vi2d &n)
	 {
		 n.x = (int)((v.x - OffSet.x) * Scale.x);
		 n.y = (int)((v.y - OffSet.y) * Scale.y);
	 }

	 // Convert coordinates from Screen Space --> World Space
	 void ScreenToWorld(const olc::vd2d& n, olc::vd2d& v)
	 {
		 v.x = (n.x) / Scale.x + OffSet.x;
		 v.y = (n.y) / Scale.y + OffSet.y;
	 }
};

int main()
{
	MendrelBrot demo;
	if (demo.Construct(1280, 720, 1, 1))
	{
		demo.Start();
	}
	return 0;
}


