#include "ComputeMethods.h"
#include <immintrin.h>
#include <future>

void frac_basic::compute(const olc::vd2d & ScreenTL, const olc::vd2d & ScreenBR, const olc::vd2d & FractalTL, const olc::vd2d & FractalBR, int maxiterations)
{
	double x_scale = (FractalBR.x - FractalTL.x) / (double(ScreenBR.x) - double(ScreenTL.x));
	double y_scale = (FractalBR.y - FractalTL.y) / (double(ScreenBR.y) - double(ScreenTL.y));

	double x_pos = FractalTL.x;
	double y_pos = FractalTL.y;

	int row_size = screenwidth;

	int n;		//iteration counter

	double cr = 0;
	double ci = 0;
	double zr = 0;
	double zi = 0;
	double re = 0;
	double im = 0;

	for (int y = ScreenTL.y; y < ScreenBR.y; ++y)
	{
		y_pos = FractalTL.y;
		ci = y_pos;
		for (int x = ScreenTL.x; x < ScreenBR.x; ++x)
		{
			cr = x_pos;
			zr = 0;
			zi = 0;
			n = 0;

			while ((zr * zr + zi * zi) < 4.0 && n < maxiterations)
			{
				re = zr * zr - zi * zi + cr;
				im = zr * zi * 2.0 + ci;
				zr = re;
				zi = im;
				++n;
			}

			iteration_vec[x + screenwidth * y] = n;
			y_pos += y_scale;			//increment to the next position in fractal space.
		}
		x_pos += x_scale;				//increment to the next position in fractal space
	}
}

void frac_basic_SIMD::compute(const olc::vd2d& ScreenTL, const olc::vd2d& ScreenBR, const olc::vd2d& FractalTL, const olc::vd2d& FractalBR, int maxiterations)
{
	const double scale_x = (FractalTL.x - FractalBR.x) / (ScreenTL.x - ScreenBR.x);
	const double scale_y = (FractalTL.y - FractalBR.y) / (ScreenTL.y - ScreenBR.y);

	//set up scale x and scale y vector
	__m256d scale_x_vec = _mm256_set1_pd(scale_x);
	__m256d scale_y_vec = _mm256_set1_pd(scale_y);

	//fractal top left vector
	__m256d frac_tl_y_vec = _mm256_set1_pd(FractalTL.y);
	__m256d frac_br_x_vec = _mm256_set1_pd(FractalTL.x);

	__m256d ci_vec, cr_vec, zi_vec, zr_vec, temp_vec;
	__m256d zi_sq_vec;	//zi^2
	__m256d zr_sq_vec;	//zr^2 

	
	__m256d cmp_cond_1; //vector to store comparison results.
	__m256i cmp_cond_2; //vector to store comparison results.
	
	__m256i n_vec;			//iteration counter vector
	__m256i store_n_vec;	//accumlate the itration counter

	__m256d z_mag_vec;		//magnitude for z
	

	//initialize constants
	__m256d two = _mm256_set1_pd(2.0);		
	__m256d four = _mm256_set1_pd(4.0);		
	__m256i one = _mm256_set1_epi64x(1);
	__m256i maxiterationsVector = _mm256_set1_epi64x(maxiterations);
	__m256d Pixels = _mm256_set_pd(0, 1, 2, 3);

	__m256d x_pos_vec = _mm256_set1_pd(FractalTL.x);
	__m256d y_pos_vec = _mm256_set1_pd(FractalTL.y);
	__m256d offset = _mm256_mul_pd(Pixels, scale_y_vec);	//convert pixel to fractal space
	__m256d jmp = _mm256_mul_pd(four, scale_y_vec);			//to jump four pixels ahead

												//for each pixel in screen space
	for (int y = (int)ScreenTL.y; y < (int)ScreenBR.y; ++y)
	{
		ci_vec = y_pos_vec;
		x_pos_vec = _mm256_add_pd(x_pos_vec, scale_x_vec);
		y_pos_vec = _mm256_add_pd(frac_tl_y_vec, offset);

		for (int x = (int)ScreenTL.x; x < (int)ScreenBR.x; x += 4)
		{

			cr_vec = x_pos_vec;

			//initilaize Z Vector
			//double ZReal = 0.0;
			zr_vec = _mm256_set1_pd(0.0);
			//double ZIm = 0.0;
			zi_vec = _mm256_set1_pd(0.0);

			//int iteration_counter = 0;
			store_n_vec = _mm256_set1_epi64x(0);

		repeat:
			//F(Z) = Z^2  + C
			//double ZTemp = (ZReal * ZReal) - (ZIm * ZIm) + CReal;
			//Z real squared and Z im squared vecotr
			zr_sq_vec = _mm256_mul_pd(zr_vec, zr_vec);
			zi_sq_vec = _mm256_mul_pd(zi_vec, zi_vec);

			//Zre^2 + Zim^2
			z_mag_vec = _mm256_add_pd(zr_sq_vec, zi_sq_vec);

			//Zre^2 + Zim^2 < 4.0
			cmp_cond_1 = _mm256_cmp_pd(z_mag_vec, four, _CMP_LT_OQ);

			temp_vec = _mm256_mul_pd(zr_vec, zr_vec);
			temp_vec = _mm256_sub_pd(temp_vec, _mm256_mul_pd(zi_vec, zi_vec));
			temp_vec = _mm256_add_pd(temp_vec, cr_vec);

			//ZIm = 2 * (ZIm * ZReal) + CIm;
			zi_vec = _mm256_mul_pd(zi_vec, zr_vec);
			zi_vec = _mm256_fmadd_pd(zi_vec, two, ci_vec);

			//ZReal = ZTemp;
			zr_vec = temp_vec;

			//maxiterations > iterations
			cmp_cond_2 = _mm256_cmpgt_epi64(maxiterationsVector, store_n_vec);

			//condition1 && condition 2
			cmp_cond_2 = _mm256_and_si256(cmp_cond_2, _mm256_castpd_si256(cmp_cond_1));

			//increment iteration ONLY IF the condition for a pixel is true
			n_vec = _mm256_and_si256(cmp_cond_2, one);

			//add 1 to the iteration counter
			//++iteration_counter;
			store_n_vec = _mm256_add_epi64(store_n_vec, n_vec);

			if ((n_vec.m256i_i64[0] == 0 && n_vec.m256i_i64[1] == 0 && n_vec.m256i_i64[2] == 0 && n_vec.m256i_i64[3] == 0))
			{
				iteration_vec[x + screenwidth * y] = int(store_n_vec.m256i_i64[3]);
				iteration_vec[x + screenwidth * (y + 1)] = int(store_n_vec.m256i_i64[2]);
				iteration_vec[x + screenwidth * (y + 2)] = int(store_n_vec.m256i_i64[1]);
				iteration_vec[x + screenwidth * (y + 3)] = int(store_n_vec.m256i_i64[0]);

				x_pos_vec = _mm256_add_pd(jmp, x_pos_vec);
			}

			else {
				goto repeat;
			}
		}
	}
}

void frac_multithread_SIMD::compute(const olc::vd2d & ScreenTL, const olc::vd2d & ScreenBR, const olc::vd2d & FractalTL, const olc::vd2d & FractalBR, int maxiterations)
{
	
	std::thread t1[thread_num];

	int ScreenWidth = (ScreenBR.x - ScreenTL.x) / thread_num;
	double FractalWidth = (FractalBR.x - FractalTL.x) / double(thread_num);

	for (int i = 0; i < thread_num; ++i)
	{
		t1[i] = std::thread(&frac_basic_SIMD::compute, this,
			olc::vd2d(ScreenTL.x + ScreenWidth * (i), ScreenTL.y),			//SCREENTL
			olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1), ScreenBR.y),			//SCREENBR
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
			maxiterations);
	}

	for (int i = 0; i < thread_num; ++i)
	{
		t1[i].join();
	}
}

void frac_threadpool_SIMD::compute(const olc::vd2d & ScreenTL, const olc::vd2d & ScreenBR, const olc::vd2d & FractalTL, const olc::vd2d & FractalBR, int maxiterations)
{
	int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
	double FractalWidth = (FractalBR.x - FractalTL.x) / double(ThreadNumbers);

	//workers = 0;

	using Task = std::packaged_task<void()>;

	for (int i = 0; i < ThreadNumbers; ++i)
	{
		Task t = Task(std::bind(&frac_basic_SIMD::compute, this,
			olc::vd2d(ScreenTL.x + ScreenWidth * (i), ScreenTL.y),			//SCREENTL
			olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1), ScreenBR.y),			//SCREENBR
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
			maxiterations));
		//pool.enqueue(std::move(t));

	}

	while (workers < thread_num) {}	//prevent screen tearing.

}

void frac_async_SIMD::compute(const olc::vd2d & ScreenTL, const olc::vd2d & ScreenBR, const olc::vd2d & FractalTL, const olc::vd2d & FractalBR, int maxiterations)
{
	

	int ScreenWidth = (ScreenBR.x - ScreenTL.x) / ThreadNumbers;
	double FractalWidth = (FractalBR.x - FractalTL.x) / double(ThreadNumbers);

	std::vector<std::future<void>> fus;

	for (int i = 0; i < thread_num; ++i) {

		fus.push_back(std::async(std::launch::async, &frac_basic_SIMD::compute, this,
			olc::vd2d(ScreenTL.x + ScreenWidth * (i), ScreenTL.y),			//SCREENTL
			olc::vd2d(ScreenTL.x + ScreenWidth * (i + 1), ScreenBR.y),			//SCREENBR
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i), FractalTL.y),		//FRACTALTL
			olc::vd2d(FractalTL.x + FractalWidth * (double)(i + 1), FractalBR.y),	//FRACTALBR
			maxiterations));
	}

	for (auto& fu : fus)
	{
		fu.wait();
	}
}


