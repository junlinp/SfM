#ifndef INTERNAL_THREAD_POOL_H_
#define INTERNAL_THREAD_POOL_H_

#include <thread>
#include <condition_variable>
#include <future>
#include <queue>

class ThreadPool {
public:
    ThreadPool(size_t max_thread = 2) : is_stop_(false) {
        auto work = [&]() {
            std::function<void()> task;
            while(true) {
                {
                    std::unique_lock<std::mutex> ulk(tasks_mutex_);
                    cv_.wait(ulk, [this](){ return !tasks_.empty() || this->is_stop_;});

                    if (is_stop_ && tasks_.empty()) {
                        break;
                    }

                    if (tasks_.size()) {
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                }
                task();
            }
        };

        for(int i = 0; i < max_thread; i++) {
            works_.emplace_back(std::thread(work));
        }
    };

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> ulk(tasks_mutex_);
            is_stop_ = true;
        }
        cv_.notify_all();
        for(std::thread& t : works_) {
            t.join();
        }
    }

    template<class Functor, class ...Args>
    auto Enqueue(Functor&& functor, Args&&... args) {
        //using Origin_Functor = std::remove_reference_t<Functor>;
        using Origin_Functor = Functor;

        using Functor_Return_Type = std::result_of_t<Origin_Functor(Args...)>;

        auto bind_functor = std::bind(std::forward<Functor>(functor), std::forward<Args>(args)...);
        auto task = std::make_shared<std::packaged_task<Functor_Return_Type()>>(bind_functor);

        std::future result_future = task->get_future();
        {
            std::unique_lock<std::mutex> ulk(tasks_mutex_);
            if (is_stop_) {
                throw std::runtime_error("Thread Pool has stop\n");
            }
            tasks_.push([task](){(*task)();});

        }
        cv_.notify_one();
        return result_future;
    }

private:
   std::vector<std::thread> works_;
   std::queue<std::function<void()>> tasks_;
   std::condition_variable cv_;
   std::mutex tasks_mutex_;
    bool is_stop_;
};
#endif // INTERNAL_THREAD_POOL_H_