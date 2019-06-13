//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_IMAGE_HPP_
#define SFM_SRC_IMAGE_HPP_
#include "keypoint.hpp"
#include "descriptor.hpp"

class image {
private:
    std::vector<std::shared_ptr<descriptor>> descriptor_ptr;
    std::vector<std::shared_ptr<keypoint>> keypoint_ptr;
    int width, height;
    std::string path;
    std::string name;
    
    double qvec[4];
    double tvec[3];

    //load from database to assign
    int camera_id;
    int image_id;

public:
    image();
    bool load_image(std::string path);
    size_t get_features_num() const;

    std::shared_ptr<descriptor> get_descriptor(int idx);
    std::shared_ptr<keypoint> get_keypoint(int idx);

    std::vector<std::shared_ptr<descriptor>> get_descriptors();
    std::vector<std::shared_ptr<keypoint>> GetKeyPoints();


    void descripotr_push_back(std::shared_ptr<descriptor> descriptor);
    void keypoint_push_back(std::shared_ptr<keypoint> keypoint);

    int get_camera_id();
    int get_image_id();

    void set_camera_id(int id);
    void set_image_id(int id);

    std::string get_name();
    void SetName(std::string name);

    double* GetQvec();
    double* GetTvec();

    int GetWidth();
    int GetHeight();

    void SetWidth(int width);
    void SetHeight(int height);

    bool equals(image* img);
    bool equals(image img);

    ~image();
};

#endif //SFM_SRC_IMAGE_HPP_
