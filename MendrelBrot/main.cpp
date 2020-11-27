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
#include "ComputeMethodsFactory.h"
#include "Renderer.h"

class MendrelBrot : public olc::PixelGameEngine
{

public:
	MendrelBrot()
	{
		sAppName = "MendrelBrot";
	};

public:

	bool OnUserCreate() override
	{	
		
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

		/*generate factory here (use 1 , 2 ,3 ,4, 5, 6)*/
		if ((GetKey(olc::Key::K1).bPressed)) {
			curr_type = 1;
			method = std::move(ComputeMethodsFactory::create(ScreenHeight(), ScreenWidth(), std::thread::hardware_concurrency(), curr_type));
		}

		if ((GetKey(olc::Key::K2).bPressed)) {
			curr_type = 2;
			method = std::move(ComputeMethodsFactory::create(ScreenHeight(), ScreenWidth(), std::thread::hardware_concurrency(), curr_type));
		}

		if (1) {
			curr_type = 3;
			method = std::move(ComputeMethodsFactory::create(ScreenHeight(), ScreenWidth(), std::thread::hardware_concurrency(), curr_type));
		}

		if ((GetKey(olc::Key::K4).bPressed)) {
		
			curr_type = 4;
			method = std::move(ComputeMethodsFactory::create(ScreenHeight(), ScreenWidth(), std::thread::hardware_concurrency(), curr_type));
		}

		if ((GetKey(olc::Key::K5).bPressed)) {
			
			curr_type = 5;
			method = std::move(ComputeMethodsFactory::create(ScreenHeight(), ScreenWidth(), std::thread::hardware_concurrency(), curr_type));
		}

		/*benchmark here (call compute)*/
		if (method) {
			auto start = std::chrono::high_resolution_clock::now();
			method->compute(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			auto end = std::chrono::high_resolution_clock::now();
			dt = end - start;

			/*render*/
			method = std::move(render.RenderFractal(std::move(method), ScreenWidth(), ScreenHeight(), *this));
			render.RenderUI(*this, dt, std::to_string(curr_type), maxiterations);
		}

		return true;
	}


private:
	std::unique_ptr<ComputeMethods> method;
	Renderer render;
	olc::vd2d MouseStartPos = { 0 ,0 };
	olc::vd2d Scale = { 1280.0 * 0.5 , 720.0 };
	olc::vd2d OffSet = { 0.0 , 0.0 };

	int curr_type;
	
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


