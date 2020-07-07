#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <immintrin.h>
#include <thread>


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
			IterationStore.emplace_back(0);
		}

		return true;
	}

	void ComputeFractal(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
	{
		const double ScaleX = (FractalTL.x - FractalBR.x) / (ScreenTL.x - ScreenBR.x);
		const double ScaleY = (FractalTL.y - FractalBR.y) / (ScreenTL.y - ScreenBR.y);

		//for each pixel in screen space
		for (int x = ScreenTL.x; x < ScreenBR.x; ++x)
		{
			for (int y = ScreenTL.y; y < ScreenBR.y; ++y)
			{
				/*C part*/
				double CReal = ((double)x * ScaleX) + FractalTL.x;
				double CIm = ((double)y * ScaleY) + FractalTL.y;

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

	void ComputeFractalSIMD(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
	{
		const double ScaleX = (FractalTL.x - FractalBR.x) / (ScreenTL.x - ScreenBR.x);
		const double ScaleY = (FractalTL.y - FractalBR.y) / (ScreenTL.y - ScreenBR.y);
		
		//set up scale x and scale y vector
		__m256d ScaleXVector = _mm256_set1_pd(ScaleX);
		__m256d ScaleYVector = _mm256_set1_pd(ScaleY);

		//fractal top left vector
		__m256d FractalTLVectorY = _mm256_set1_pd(FractalTL.y);
		__m256d FractalTLVectorX = _mm256_set1_pd(FractalTL.x);

		__m256d CImVector, CReVector, ZImVector, ZReVector, ZTempVector;
		__m256d ZImSquared, ZReSquared;
		
		/*vector to store comparison results.*/
		__m256d cmp_cond_1;
		__m256i cmp_cond_2;
		__m256i CounterVector;		//counter vector
		
		__m256d ZMagnitude;	//ZMagnitude t check for bounds

		__m256d two, four;	//constant vectors
		__m256i one, maxiterationsVector, IterationVector;

		/*initialize constants*/
		two = _mm256_set1_pd(2.0);		//for computing fractal
		four = _mm256_set1_pd(4.0);		//to check while loop condition.
		one = _mm256_set1_epi64x(1);
		maxiterationsVector = _mm256_set1_epi64x(maxiterations);

		__m256d YValue, XValue;

		//for each pixel in screen space
	for (int x = ScreenTL.y; x < ScreenBR.x; ++x)
		{
			XValue = _mm256_set1_pd((double)x);

			for (int y = ScreenTL.y; y < ScreenBR.y; y += 4)
			{
				/*what we CAN do , is compute 4 doubles at ONE go with mm256. we could calculate for 4 complex numbers at a time*/
				
				/*C part*/
				
				//double CIm = (y * ScaleY) + FractalTL.y;
				/*imaginary C computation*/
				YValue = _mm256_set_pd((double)y, (double)(y + 1), (double)(y + 2), (double)(y + 3));
				CImVector = _mm256_fmadd_pd(YValue, ScaleYVector,FractalTLVectorY);

				//double CReal = (x * ScaleX) + FractalTL.x;
				/*Real C computation*/
				CReVector = _mm256_fmadd_pd(XValue, ScaleXVector, FractalTLVectorX);

				/*initilaize Z Vector*/
				//double ZReal = 0.0;
				ZReVector = _mm256_set1_pd(0.0);
				//double ZIm = 0.0;
				ZImVector = _mm256_set1_pd(0.0);

				//int iteration_counter = 0;
				IterationVector = _mm256_set1_epi64x(0);

			repeat:
					/*F(Z) = Z^2  + C*/
					//double ZTemp = (ZReal * ZReal) - (ZIm * ZIm) + CReal;
					//Z real squared and Z im squared vecotr
					ZReSquared = _mm256_mul_pd(ZReVector, ZReVector);
					ZImSquared = _mm256_mul_pd(ZImVector, ZImVector);

					/*Zre^2 + Zim^2*/
					ZMagnitude = _mm256_add_pd(ZReSquared, ZImSquared);

					/*Zre^2 + Zim^2 < 4.0*/
					cmp_cond_1 = _mm256_cmp_pd(ZMagnitude, four, _CMP_LT_OQ);

					ZTempVector = _mm256_mul_pd(ZReVector , ZReVector);	
					ZTempVector = _mm256_sub_pd(ZTempVector, _mm256_mul_pd(ZImVector, ZImVector));
					ZTempVector = _mm256_add_pd(ZTempVector, CReVector);
					
					//ZIm = 2 * (ZIm * ZReal) + CIm;
					ZImVector = _mm256_mul_pd(ZImVector , ZReVector);
					ZImVector = _mm256_fmadd_pd(ZImVector,two, CImVector);
					
					//ZReal = ZTemp;
					ZReVector = ZTempVector;

					/*maxiterations > \iterations*/
					cmp_cond_2 = _mm256_cmpgt_epi64(maxiterationsVector,IterationVector);

					/*condition1 && condition 2*/
					cmp_cond_2 = _mm256_and_si256(cmp_cond_2, _mm256_castpd_si256(cmp_cond_1));

					/*increment iteration ONLY IF the condition for a pixel is true*/
					CounterVector = _mm256_and_si256(cmp_cond_2, one);

					/*add 1 to the iteration counter*/
					//++iteration_counter;
					IterationVector = _mm256_add_epi64(IterationVector, CounterVector);

					if ((CounterVector.m256i_i64[0] == 0 && CounterVector.m256i_i64[1] == 0 && CounterVector.m256i_i64[2] == 0 && CounterVector.m256i_i64[3] == 0))
					{

						IterationStore[x + ScreenWidth() * y] = int(IterationVector.m256i_i64[3]);
						IterationStore[x + ScreenWidth() * (y + 1)] = int(IterationVector.m256i_i64[2]);
						IterationStore[x + ScreenWidth() * (y + 2)] = int(IterationVector.m256i_i64[1]);
						IterationStore[x + ScreenWidth() * (y + 3)] = int(IterationVector.m256i_i64[0]);
					}
					else {
						//continue the while loop if any outputs of the counter are not zero.
						goto repeat;
					}
			}
		}
	}

	void ComputeFractal_4_Threads(olc::vd2d& ScreenTL, olc::vd2d& ScreenBR, olc::vd2d& FractalTL, olc::vd2d& FractalBR, int maxiterations)
	{
		static constexpr uint8_t ThreadNumbers = 2;
		std::thread t1[ThreadNumbers];		

		int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
		double FractalWidth = (FractalBR.x - FractalTL.x) / (double)ThreadNumbers;
		
		for (int i = 0; i < ThreadNumbers; ++i)
		{
			t1[i] = std::thread(&MendrelBrot::ComputeFractalSIMD,this,
				olc::vd2d(ScreenTL.x +  ScreenWidth * (i) , ScreenTL.y),			//SCREENTL
				olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1) , ScreenBR.y),	//SCREENBR
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),
				maxiterations);
		}

		for (int i = 0; i < ThreadNumbers; ++i)
		{
			t1[i].join();
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
				static constexpr float a = 0.1f;
				Draw(x, y, olc::PixelF(0.5f * sinf(a * n) + 0.5f, 0.5f * sinf(a * n + 2.094f) + 0.5f, 0.5f * sinf(a * n + 4.188f) + 0.5f));
			}
		}
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		//top left and bottom right of screenspace
		olc::vd2d ScreenTL = { 0 , 0 };
		olc::vd2d ScreenBR = { 1280, 720 };

		//top elft and bottom right of mendrelbrot space
		olc::vd2d FractalTL = { -2.0f , -1.0f };
		olc::vd2d FractalBR = { 1.0f , 1.0f };

		/*map the screen space to world space*/
		ScreenToWorld(ScreenTL, FractalTL);
		ScreenToWorld(ScreenBR, FractalBR);

		//compute performance
		


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

		if (GetKey(olc::Key::Z).bPressed)
		{
			
			SimulationType += 1;
			SimulationType = SimulationType % SimulationNumbers;
		}

		auto StartTime = std::chrono::high_resolution_clock::now();
		switch (SimulationType)
		{
		case 0:
			SimulationTypeString = "naive algorithm";
			ComputeFractal(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;

		case 1:
			SimulationTypeString = "SIMD";
			ComputeFractalSIMD(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;


		case 2:
			SimulationTypeString = "4 threads";
			ComputeFractal_4_Threads(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;
		}

		auto EndTime = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> TimeTaken = EndTime - StartTime;

		//render the fractal (not included in computation)
		RenderFractal();

		//print UI
		DrawString(0, 0, "Simulation Type: " + SimulationTypeString, olc::BLACK, 3);
		DrawString(0, 30, "Time Taken: " + std::to_string(TimeTaken.count()) + "s", olc::BLACK, 3);
		DrawString(0, 60, "Iterations: " + std::to_string(maxiterations), olc::BLACK, 3);
	
		return true;
	}

private:
	olc::vd2d MouseStartPos;
	olc::vd2d Scale = { 1280.0 * 0.5 , 720.0 };
	olc::vd2d OffSet = { 0.0 , 0.0 };

	int SimulationType = 0;			//specifiy which type of simulation is being simulated.
	int SimulationNumbers = 3;		//number of simualtions avaiable 
	std::string SimulationTypeString;

	int maxiterations = 50;

private:
	 //stores the iteration counter for the fractal (purpose for colouring the fractal)
	
	std::vector<int> IterationStore;;
	
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