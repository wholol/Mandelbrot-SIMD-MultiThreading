#include "ThreadPool.h"

ThreadPool::ThreadPool()
{
}

ThreadPool::ThreadPool(int numthreads)
{
	start(numthreads);
}

void ThreadPool::enqueue(Task task)
{
	{
		//get lock, unlocks when it goes out of scope
		std::unique_lock<std::mutex> m(eventmutex);
		q.push_back(std::move(task));
	}

	eventvar.notify_one();
}

ThreadPool::~ThreadPool()
{
	stop();
}

void ThreadPool::start(int numThreads)
{
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

void ThreadPool::stop()
{
	std::unique_lock<std::mutex> lock(eventmutex);
	bStop = true;
	eventvar.notify_all();

	for (auto& t : threads)
	{
		t.join();
	}
}
