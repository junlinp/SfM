//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_UNITTEST_HPP_
#define SFM_SRC_UNITTEST_HPP_
#include <gtest/gtest.h>

#include "internal/thread_pool.hpp"
#include "ransac.hpp"

TEST(ThreadPool, Enqueue) {
    auto functor = [](int a) {
        return 2 * a;
    };
    ThreadPool threadpool;
    for(int i = 0; i < 1024; i++) {
        std::future<int> res = threadpool.Enqueue(functor, i);
        EXPECT_EQ(2 * i, res.get());
    }
}

TEST(Ransac, Fit_Line) {

}

#endif //SFM_SRC_UNITTEST_HPP_
