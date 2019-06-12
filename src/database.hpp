//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_DATABASE_HPP_
#define SFM_SRC_DATABASE_HPP_
#include "sqlite3.h"
#include <string>
#include "descriptor.hpp"
#include "keypoint.hpp"
#include "camera_model.hpp"
#include <vector>
#include "image.hpp"

class database {

 private:
  std::string name;
  sqlite3 *sqlite_handle;
 public:
  explicit database(std::string name);
  bool created();
  bool create();
  template<class Camera_Model>
  bool insert_image(std::shared_ptr<image> img,
                    Camera_Model camera_model) {
    if (img == nullptr) {
      return false;
    }
    const auto& kp = img->get_keypoints();
    const auto& desc = img->get_descriptors();
    const std::string name = img->get_name();
    // simple
    const char *sql = "INSERT INTO t_camera("
                      "camera_model,"
                      "camera_param_num,"
                      "camera_param"
                      ")VALUES(?, ?, ?);";
    sqlite3_stmt *stmt = nullptr;

    const char *msg = nullptr;
    sqlite3_exec(sqlite_handle, "begin;", nullptr, nullptr, nullptr);
    int rc = sqlite3_prepare_v2(this->sqlite_handle, sql, strlen(sql), &stmt, &msg);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "Error : %s\n", msg);
      return false;
    }

    sqlite3_bind_int(stmt, 1, camera_model_trait<Camera_Model>::model_id);
    // TODO:get the camera meta data
    sqlite3_bind_int(stmt, 2, camera_model_trait<Camera_Model>::param_num);
    sqlite3_bind_blob(stmt,
                      3,
                      (void*)camera_model_trait<Camera_Model>::initial_param,
                      camera_model_trait<Camera_Model>::param_num
                          * sizeof(typename camera_model_trait<Camera_Model>::param_type),
                      nullptr);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    sqlite3_int64 camera_id = sqlite3_last_insert_rowid(this->sqlite_handle);

    if (rc != SQLITE_DONE) {
      return false;
    }
    sql = "INSERT INTO t_image("
          "qw, qx, qy,qz, tx, ty,tz, camera_id, name, width, height)"
          "VALUES(?,?,?,?,?,?,?,?,?, ?, ?)";
    sqlite3_prepare_v2(this->sqlite_handle, sql, strlen(sql), &stmt, nullptr);
    sqlite3_bind_double(stmt, 1, 1.0);
    sqlite3_bind_double(stmt, 2, 0.0);
    sqlite3_bind_double(stmt, 3, 0.0);
    sqlite3_bind_double(stmt, 4, 0.0);
    sqlite3_bind_double(stmt, 5, 0.0);
    sqlite3_bind_double(stmt, 6, 0.0);
    sqlite3_bind_double(stmt, 7, 0.0);

    sqlite3_bind_int(stmt, 8, camera_id);
    sqlite3_bind_text(stmt, 9, name.c_str(), int(name.size()), nullptr);
    sqlite3_bind_int(stmt, 10, img->GetWidth());
    sqlite3_bind_int(stmt, 11, img->GetHeight());

    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    sqlite3_int64 image_id = sqlite3_last_insert_rowid(this->sqlite_handle);

    sql = "INSERT INTO t_keypoint("
          "image_id, idx, x, y)"
          "VALUES(?, ?, ?, ?)";
    sqlite3_prepare_v2(this->sqlite_handle, sql, strlen(sql), &stmt, nullptr);
    for (int i = 0; i < kp.size(); i++) {

      sqlite3_bind_int(stmt, 1, int(image_id));
      sqlite3_bind_int(stmt, 2, i);
      sqlite3_bind_double(stmt, 3, kp[i]->x);
      sqlite3_bind_double(stmt, 4, kp[i]->y);
      sqlite3_step(stmt);
      sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);

    sql = "INSERT INTO t_descript("
          "image_id, idx, width, bytes)"
          "VALUES(?, ?, ?, ?)";
    sqlite3_prepare_v2(sqlite_handle, sql, strlen(sql), &stmt, nullptr);
    for (int i = 0; i < desc.size(); i++) {
      sqlite3_bind_int(stmt, 1, int(image_id));
      sqlite3_bind_int(stmt, 2, i);
      sqlite3_bind_int(stmt, 3, desc[i]->size());
      sqlite3_bind_blob(stmt, 4, (const void*) desc[i]->data(), desc[i]->size(), nullptr);
      sqlite3_step(stmt);
      sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    sqlite3_exec(sqlite_handle, "commit;", nullptr, nullptr, nullptr);
    return true;
  }

  // TODO: load camera load image to match and insert the result into the database


  std::shared_ptr<image> GetImageById(int id);

  std::vector<std::shared_ptr<image>> GetImages();

  ~database();
};

#endif //SFM_SRC_DATABASE_HPP_
