#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <complex>

class MendrelBrot : public olc::PixelGameEngine
{
public:
	MendrelBrot()
	{
		sAppName = "MendrelBrot";
	}
public:


	bool OnUserCreate() override
	{	
		IterationStore.reserve(ScreenHeight() * ScreenWidth());
		for (int i = 0; i < ScreenHeight() * ScreenWidth(); ++i)
		{
			IterationStore.push_back(0);
		}
		return true;
	}

	void ComputeFractal(olc::vd2d& ScreenTL, olc::vd2d& ScreenBR, olc::vd2d& FractalTL, olc::vd2d& FractalBR, int maxiterations)
	{
		const double ScaleX = (FractalTL.x - FractalBR.x) / (ScreenTL.x - ScreenBR.x);
		const double ScaleY = (FractalTL.y - FractalBR.y) / (ScreenTL.y - ScreenBR.y);

		//for each pixel in screen space
		for (int x = ScreenTL.x; x < ScreenBR.x; ++x)
		{
			for (int y = ScreenTL.y; y < ScreenBR.y; ++y)
			{
				/*C part*/
				double CReal = (x * ScaleX) + FractalTL.x;
				double CIm = (y * ScaleY) + FractalTL.y;

				/*initialize Z*/
				double ZReal = 0.0;
				double ZIm = 0.0;

				int iteration_counter = 0;

				while ((ZReal*ZReal + ZIm * ZIm) < 4.0 && iteration_counter < maxiterations)
				{
					/*F(Z) = Z^2  + C*/
					double ZTemp = (ZReal * ZReal) - (ZIm * ZIm) + CReal;
					ZIm = 2 * (ZIm * ZReal) + CIm;
					ZReal = ZTemp;
					++iteration_counter;
				}

				IterationStore[x + ScreenWidth() * y] = iteration_counter;
			}
		}
	}

	//render fractal with some colours.
	void RenderFractal()
	{
		for (int x = 0; x < ScreenWidth(); ++x)
		{
			for (int y = 0; y < ScreenHeight(); ++y)
			{
				float n = (float)IterationStore[x + ScreenWidth() * y];
				const float a = 0.1f;
				Draw(x, y, olc::PixelF(0.5f * sinf(a * n) + 0.5f, 0.5f * sinf(a * n + 2.094f) + 0.5f, 0.5f * sinf(a * n + 4.188f) + 0.5f));
			}
		}
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
		}
		

	   //top left and bottom right of screenspace
		olc::vd2d ScreenTL = { 0 , 0 };
		olc::vd2d ScreenBR = { 1280, 720 };

		//top elft and bottom right of mendrelbrot space
		olc::vd2d FractalTL = { -2.0f , 1.0f };
		olc::vd2d FractalBR = { 1.0f , -1.0f };

		/*map the screen space to world space*/
		ScreenToWorld(ScreenTL, FractalTL);
		ScreenToWorld(ScreenBR, FractalBR);

		//compute performance
		auto StartTime = std::chrono::high_resolution_clock::now();

		ComputeFractal(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);

		auto EndTime = std::chrono::high_resolution_clock::now();
		
		std::chrono::duration<double> TimeTaken = EndTime - StartTime;

		//std::cout << TimeTaken.count() << std::endl;

		//render the fractal (not included in computation)
		RenderFractal();
		
		DrawString(0, 30, "Time Taken: " + std::to_string(TimeTaken.count()) + "s", olc::BLACK, 3);
		DrawString(0, 60, "Iterations: " + std::to_string(maxiterations), olc::BLACK, 3);
	
		return true;
	}

private:
	olc::vd2d MouseStartPos;
	olc::vd2d Scale = { 1280.0 / 2.0, 720.0 };
	olc::vd2d OffSet = { 0.0 , 0.0 };

	int maxiterations = 50;

private:
	 //stores the iteration counter for the fractal (purpose for colouring the fractal)
	 std::vector<int> IterationStore;

	 void WorldToScreen(const olc::vd2d& v, olc::vi2d &n)
	 {
		 n.x = (int)((v.x - OffSet.x) * Scale.x);
		 n.y = (int)((v.y - OffSet.y) * Scale.y);
	 }

	 // Convert coordinates from Screen Space --> World Space
	 void ScreenToWorld(const olc::vi2d& n, olc::vd2d& v)
	 {
		 v.x = (double)(n.x) / Scale.x + OffSet.x;
		 v.y = (double)(n.y) / Scale.y + OffSet.y;
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