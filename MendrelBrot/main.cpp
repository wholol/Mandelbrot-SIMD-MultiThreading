#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <immintrin.h>
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>
#include <future>

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

		//init threadpool

		return true;
	}

	void frac_precalc(const olc::vi2d& pix_tl, const olc::vi2d& pix_br, const olc::vd2d& frac_tl, const olc::vd2d& frac_br, const int iterations)
	{
		double x_scale = (frac_br.x - frac_tl.x) / (double(pix_br.x) - double(pix_tl.x));
		double y_scale = (frac_br.y - frac_tl.y) / (double(pix_br.y) - double(pix_tl.y));

		double x_pos = frac_tl.x;
		double y_pos = frac_tl.y;

		int y_offset = 0;
		int row_size = ScreenWidth();

		int x, y, n;

		double cr = 0;
		double ci = 0;
		double zr = 0;
		double zi = 0;
		double re = 0;
		double im = 0;

		for (x = pix_tl.x; x < pix_br.x; x++)
		{
			y_pos = frac_tl.y;
			cr = x_pos;
			for (y = pix_tl.y; y < pix_br.y; y++)
			{
				ci = y_pos;
				zr = 0;
				zi = 0;

				n = 0;
				while ((zr * zr + zi * zi) < 4.0 && n < iterations)
				{
					re = zr * zr - zi * zi + cr;
					im = zr * zi * 2.0 + ci;
					zr = re;
					zi = im;
					n++;
				}

				IterationStore[x + ScreenWidth() * y] = n;
				y_pos += y_scale;
			}

			x_pos += x_scale;
		}
	}

	//algorithm with SIMD (pre calculate version)
	void frac_precalc_SIMD(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
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
		__m256i CounterVector;		

		__m256d ZMagnitude;	
		__m256i IterationVector;

		/*initialize constants*/
		__m256d two = _mm256_set1_pd(2.0);		//for computing fractal
		__m256d four = _mm256_set1_pd(4.0);		//to check while loop condition.
		__m256i one = _mm256_set1_epi64x(1);
		__m256i maxiterationsVector = _mm256_set1_epi64x(maxiterations);
		__m256d Pixels = _mm256_set_pd(0, 1, 2, 3);
		
		__m256d XPos = _mm256_set1_pd(FractalTL.x);
		__m256d YPos = _mm256_set1_pd(FractalTL.y);
		__m256d offset = _mm256_mul_pd(Pixels, ScaleYVector);	//convrt pixel to fractal space
		__m256d jmp = _mm256_mul_pd(four , ScaleYVector);	//to jump four pixels ahead
		
													//for each pixel in screen space
		for (int x = (int)ScreenTL.x; x < (int)ScreenBR.x; ++x)
		{
			CReVector = XPos;
			XPos = _mm256_add_pd(XPos, ScaleXVector);
			YPos = _mm256_add_pd(FractalTLVectorY, offset);

			for (int y = (int)ScreenTL.y; y < (int)ScreenBR.y; y += 4)
			{
				
				CImVector = YPos;
				
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

					ZTempVector = _mm256_mul_pd(ZReVector, ZReVector);
					ZTempVector = _mm256_sub_pd(ZTempVector, _mm256_mul_pd(ZImVector, ZImVector));
					ZTempVector = _mm256_add_pd(ZTempVector, CReVector);

					//ZIm = 2 * (ZIm * ZReal) + CIm;
					ZImVector = _mm256_mul_pd(ZImVector, ZReVector);
					ZImVector = _mm256_fmadd_pd(ZImVector, two, CImVector);

					//ZReal = ZTemp;
					ZReVector = ZTempVector;

					/*maxiterations > \iterations*/
					cmp_cond_2 = _mm256_cmpgt_epi64(maxiterationsVector, IterationVector);

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
						YPos = _mm256_add_pd(jmp, YPos);
					}

					else {
						goto repeat;
					}
			}
		}
	}

	//normal algorithm
	void frac_basic(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
	{
		const double ScaleX = (FractalBR.x - FractalTL.x) / (ScreenBR.x - ScreenTL.x);
		const double ScaleY = (FractalBR.y - FractalTL.y) / (ScreenBR.y - ScreenTL.y);
	
		//for each pixel in screen space
		for (int x = ScreenTL.x; x < (int)ScreenBR.x; ++x)
		{
			for (int y = ScreenTL.y; y < (int)ScreenBR.y; ++y)
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

				IterationStore[x + ScreenWidth()* y] = iteration_counter;
	
			}
		}
	}

	//algorithm with SIMD
	void frac_basic_SIMD(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
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
	for (int x = (int)ScreenTL.x; x < (int)ScreenBR.x; ++x)
		{
			XValue = _mm256_set1_pd((double)x);

			for (int y = (int)ScreenTL.y; y < (int)ScreenBR.y; y += 4)
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

	void frac_multithread(olc::vd2d& ScreenTL, olc::vd2d& ScreenBR, olc::vd2d& FractalTL, olc::vd2d& FractalBR, int maxiterations)
	{
		static constexpr uint8_t ThreadNumbers = 32;	//use std::thread::hardware_concurrency() to determine max threads.
		
		std::thread t1[ThreadNumbers];		

		int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
		double FractalWidth = (FractalBR.x - FractalTL.x) / double(ThreadNumbers);
		
		for (int i = 0; i < ThreadNumbers; ++i)
		{
			t1[i] = std::thread( &MendrelBrot::frac_precalc_SIMD,this,
			olc::vd2d(ScreenTL.x + ScreenWidth * (i) , ScreenTL.y),			//SCREENTL
			olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1) , ScreenBR.y),			//SCREENBR
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
			maxiterations );
		}

		for (int i = 0; i < ThreadNumbers; ++i)
		{
			t1[i].join();
		}
	}


	void frac_threadpool(olc::vd2d& ScreenTL, olc::vd2d& ScreenBR, olc::vd2d& FractalTL, olc::vd2d& FractalBR, int maxiterations)
	{
		 uint8_t ThreadNumbers = std::thread::hardware_concurrency();	//use std::thread::hardware_concurrency() to determine max threads.

		int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
		double FractalWidth = (FractalBR.x - FractalTL.x) / double(ThreadNumbers);
		
		for (int i = 0; i < ThreadNumbers; ++i)
		{
	
			using Task = std::packaged_task<void()>;

			Task t(std::bind(&MendrelBrot::frac_precalc_SIMD, this,
				olc::vd2d(ScreenTL.x + ScreenWidth * (i), ScreenTL.y),			//SCREENTL
				olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1), ScreenBR.y),			//SCREENBR
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
				maxiterations));
			
			pool.enqueue(std::move(t));

		}
	}

	void frac_async(olc::vd2d& ScreenTL, olc::vd2d& ScreenBR, olc::vd2d& FractalTL, olc::vd2d& FractalBR, int maxiterations)
	{
		uint8_t ThreadNumbers = std::thread::hardware_concurrency();	//use std::thread::hardware_concurrency() to determine max threads.

		int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
		double FractalWidth = (FractalBR.x - FractalTL.x) / double(ThreadNumbers);

		for (int i = 0; i < ThreadNumbers; ++i) {
			
			std::future<void> v = std::async(std::launch::async, &MendrelBrot::frac_precalc_SIMD, this,
				olc::vd2d(ScreenTL.x + ScreenWidth * (i), ScreenTL.y),			//SCREENTL
				olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1), ScreenBR.y),			//SCREENBR
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
				olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
				maxiterations);
		}
	}

	//render fractal with some colours.
	void frac_render()
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

		//top left and bottom right of screenspace
		olc::vd2d ScreenTL = { 0 , 0 };
		olc::vd2d ScreenBR = { 1280, 720 };

		//top elft and bottom right of mendrelbrot space
		olc::vd2d FractalTL = { 0.0f , 0.0f };
		olc::vd2d FractalBR = { 3.0f , 2.0f };

		/*map the screen space to world space*/
		ScreenToWorld(ScreenTL, FractalTL);
		ScreenToWorld(ScreenBR, FractalBR);

		auto StartTime = std::chrono::high_resolution_clock::now();
		switch (SimulationType)
		{
		case 0:
			SimulationTypeString = "naive algorithm";
			frac_basic(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;

		case 1:
			SimulationTypeString = "SIMD";
			frac_basic_SIMD(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;

		case 2:
			SimulationTypeString = "multithreaded + SIMD";
			frac_multithread(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;

		case 3:
			SimulationTypeString = "threadpool + SIMD";
			frac_threadpool(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
			break;

		case 4:
			SimulationTypeString = "async + SIMD";
			frac_async(ScreenTL, ScreenBR, FractalTL, FractalBR, maxiterations);
		}

		auto EndTime = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> TimeTaken = EndTime - StartTime;

		//render the fractal (not included in computation)
		frac_render();

		//print UI
		DrawString(0, 0, "Simulation Type: " + SimulationTypeString, olc::BLACK, 3);
		DrawString(0, 30, "Time Taken: " + std::to_string(TimeTaken.count()) + "s", olc::BLACK, 3);
		DrawString(0, 60, "Iterations: " + std::to_string(maxiterations), olc::BLACK, 3);
		
		return true;
	}


private:
	/*threadpool method*/
	class ThreadPool
	{
	public:
		ThreadPool(int numThreads)
		{
			start(numThreads);
		};

		~ThreadPool()
		{
			stop();
		}

		using Task = std::packaged_task<void()>;

		
		void enqueue(Task task)
		{
			{
				//get lock, unlocks when it goes out of scope
				std::unique_lock<std::mutex> m(eventmutex);
				q.emplace(std::move(task));
			}

			eventvar.notify_one();
		}

	private:

		std::vector<std::thread> threads;
		std::condition_variable eventvar;
		std::mutex eventmutex;
		std::queue<Task> q;
		bool bStop = false;

		void start(int numThreads) {
			for (int i = 0; i < numThreads; ++i)
			{
				threads.emplace_back([=] {
					while (true)
					{
						Task t;

						{
							std::unique_lock<std::mutex> m(eventmutex);

							eventvar.wait(m, [=] {return bStop || !q.empty(); });

							if (bStop && q.empty()) break;

							t = std::move(q.front());
							
							q.pop();
							
						}
						t();
					}

				});
			}
		}


		void stop()
		{
			std::unique_lock<std::mutex> lock(eventmutex);
			bStop = true;
			eventvar.notify_all();

			for (auto& t : threads)
			{
				t.join();
			}
		}

	};

private:
	
	ThreadPool pool = ThreadPool(std::thread::hardware_concurrency());
	olc::vd2d MouseStartPos = { 0 ,0 };
	olc::vd2d Scale = { 1280.0 * 0.5 , 720.0 };
	olc::vd2d OffSet = { 0.0 , 0.0 };

	int SimulationType = 0;			//specifiy which type of simulation is being simulated.
	int SimulationNumbers = 5;		//number of simualtions avaiable 
	std::string SimulationTypeString;

	int maxiterations = 50;
	 //stores the iteration counter for the fractal (purpose for colouring the fractal)
	
	std::vector<int> IterationStore;
	
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