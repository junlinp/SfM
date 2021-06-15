#ifndef SFM_DATA_H_
#define SFM_DATA_H_

#include <string>
#include <map>

using IndexT = int64_t;

struct View {
    std::string image_path;
};

struct SfMData {
    std::map<IndexT, View> views;
};

#endif // SFM_DATA_H_