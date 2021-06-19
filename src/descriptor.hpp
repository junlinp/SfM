//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_DESCRIPTOR_HPP_
#define SFM_SRC_DESCRIPTOR_HPP_
#include <vector>
#include <array>

class Descriptors
{

private:
    std::vector<float> raw_data;

public:
    Descriptors() = default;

    size_t size() {
      return raw_data.size() / 128;
    }

    float* data() {
      return raw_data.data();
    }

    void push_back(std::array<float, 128>& data) {
      std::copy(data.begin(), data.end(), std::back_inserter(raw_data));
    }

};

#endif //SFM_SRC_DESCRIPTOR_HPP_
