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
	
	ThreadPool(int numThreads);

	using Task = std::packaged_task<void()>;
	void enqueue(Task task);

	~ThreadPool();

private:

	std::vector<std::thread> threads;
	std::condition_variable eventvar;
	std::mutex eventmutex;
	std::deque<Task> q;
	bool bStop = false;

	void start(int numThreads);

	void stop();
};
