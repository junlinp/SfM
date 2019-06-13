//
// Created by junlinp on 2019-06-09.
//

#include "image.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "feature_extractor.hpp"

bool image::load_image(std::string path)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
        fprintf(stderr, "Load image %s error\n", path.c_str());
        return false;
    }

    this->width = img.cols;
    this->height = img.rows;

    size_t rindex = path.rfind("/");
    this->name = path.substr(rindex + 1);
    this->path = path;

    // TODO: feature extract
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    std::shared_ptr<feature_extractor> ptr_fe = std::make_shared<feature_extractor>();
    ptr_fe->extract(img, kp, desc);

    for (const auto &k : kp)
    {

        auto ptr_k = std::make_shared<keypoint>();
        ptr_k->x = k.pt.x;
        ptr_k->y = k.pt.y;

        keypoint_ptr.emplace_back(ptr_k);
    }

    for (size_t i = 0; i < desc.rows; i++)
    {
        const cv::Mat row = desc.row(i);
        descriptor_ptr.emplace_back(std::make_shared<descriptor>(row));
    }
    return true;
}
image::image() {
  image_id = -1;
  camera_id = -1;
}
image::~image() {

}
size_t image::get_features_num() const {
  return descriptor_ptr.size();
}
std::shared_ptr<descriptor> image::get_descriptor(int idx) {
  if (idx < 0 || idx >= descriptor_ptr.size())
    return std::shared_ptr<descriptor>(nullptr);
  return descriptor_ptr[idx];
}
std::shared_ptr<keypoint> image::get_keypoint(int idx) {
  if (idx < 0 || idx >= keypoint_ptr.size())
    return std::shared_ptr<keypoint>(nullptr);
  return keypoint_ptr[idx];
}
int image::get_camera_id() {
  return this->camera_id;
}
int image::get_image_id() {
  return this->image_id;
}
void image::set_camera_id(int id) {
  this->camera_id = id;
}
void image::set_image_id(int id) {
  this->image_id = id;
}
double *image::GetQvec() {
  return qvec;
}
double *image::GetTvec() {
  return tvec;
}
std::string image::get_name() {
  return name;
}
std::vector<std::shared_ptr<descriptor>> image::get_descriptors() {
  return descriptor_ptr;
}
std::vector<std::shared_ptr<keypoint>> image::GetKeyPoints() {
  return keypoint_ptr;
}
int image::GetWidth() {
  return width;
}
int image::GetHeight() {
  return height;
}
void image::SetWidth(int width) {
  if (width >= 0)
    this->width = width;
  else {
    fprintf(stderr, "Set a width with a value less then 0\n");
  }
}
void image::SetHeight(int height) {
  if (height >= 0) {
    this->height = height;
  } else {
    fprintf(stderr, "Set a height with a value less then 0\n");
  }

}
void image::SetName(std::string name) {
  this->name = name;
}
bool image::equals(image *img) {
  if (img == nullptr)
    return false;

  bool res = this->name == img->name;
  return true;

}
void image::descripotr_push_back(std::shared_ptr<descriptor> descriptor) {
  this->descriptor_ptr.push_back(descriptor);
}
void image::keypoint_push_back(std::shared_ptr<keypoint> keypoint) {
  this->keypoint_push_back(keypoint);

}
