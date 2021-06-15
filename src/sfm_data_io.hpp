#ifndef SFM_DATA_IO_H_
#define SFM_DATA_IO_H_
#include "sfm_data.hpp"

bool Save(const SfMData& sfm_data, const std::string path);
bool Load(SfMData& sfm_data, const std::string path);

#endif  // SFM_DATA_IO_H_