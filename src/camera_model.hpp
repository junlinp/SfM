//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_CAMERA_MODEL_HPP_
#define SFM_SRC_CAMERA_MODEL_HPP_
const float ones[] = { 1.0f };
template <class T>
struct camera_model_trait{
  static const int model_id = 1;
  static const int param_num = 1;
  constexpr static const float* const initial_param = ones;
  typedef float param_type;
};
enum {
  simple_pinhole_camera = 1,
};

class simple_pinhole_camera_model {
 public:
  double* getParam();
  bool WorldToImage(double x, double y, double z, double& u, double &v, double* camera_param);
};

#endif //SFM_SRC_CAMERA_MODEL_HPP_
