#pragma once
#include <mutex>
#include <condition_variable>
#include <deque>
#include <future>

class ThreadPool
{
public:
	
	std::atomic<int> worker;

	ThreadPool();
	
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
			q.push_back(std::move(task));
		}

		eventvar.notify_one();
	}

private:

	std::vector<std::thread> threads;
	std::condition_variable eventvar;
	std::mutex eventmutex;
	std::mutex count;
	std::deque<Task> q;
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

						q.pop_front();

					}
					t();
					worker++;
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
