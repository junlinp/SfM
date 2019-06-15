//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_UNITTEST_HPP_
#define SFM_SRC_UNITTEST_HPP_
#include <gtest/gtest.h>

#include "database.hpp"
#include "image.hpp"
#include "sqlite3.h"
#include <memory>

TEST(DATABASE, Create_Database) {
  std::shared_ptr<database> ptr_database = std::make_shared<database>("database.db");
  bool rc = ptr_database->create();
  ASSERT_TRUE(rc);
  system("rm database.db");
}
TEST(DATABASE, GetImageFromDataBase) {
  std::shared_ptr<database> ptr_database = std::make_shared<database>("database.db");
  bool rc = ptr_database->create();
  ASSERT_TRUE(rc);

  std::shared_ptr<image> img = std::make_shared<image>();
  rc = img->load_image("../unittest/Img/100_7100.JPG");
  ASSERT_TRUE(rc);

  rc = ptr_database->insert_image(img, simple_pinhole_camera_model());
  ASSERT_TRUE(rc);

  auto img_load = ptr_database->GetImageById(1);
  ASSERT_EQ(img->get_name(), img_load->get_name());

  ASSERT_EQ(img->get_descriptors().size(), img_load->get_descriptors().size());
  ASSERT_EQ(img->GetKeyPoints().size(), img_load->get_descriptors().size());

  system("rm database.db");
}

TEST(DATABASE, InsertImage) {
  std::shared_ptr<database> ptr_database = std::make_shared<database>("database.db");
  bool rc = ptr_database->create();
  ASSERT_TRUE(rc);

  std::shared_ptr<image> img = std::make_shared<image>();
  rc = img->load_image("../unittest/Img/100_7100.JPG");
  ASSERT_TRUE(rc);

  rc = ptr_database->insert_image(img, simple_pinhole_camera_model());
  ASSERT_TRUE(rc);

  sqlite3 *sqlite_conn = nullptr;

  int r = sqlite3_open("database.db", &sqlite_conn);

  ASSERT_EQ(r, SQLITE_OK);

  sqlite3_stmt *stmt = nullptr;
  r = sqlite3_prepare_v2(sqlite_conn, "SELECT count(*) from t_camera", -1, &stmt, nullptr);
  ASSERT_EQ(r, SQLITE_OK);
  sqlite3_step(stmt);
  int count_line = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  ASSERT_EQ(count_line, 1);

  r = sqlite3_prepare_v2(sqlite_conn,
                         "SELECT camera_model, camera_param_num, camera_param from t_camera",
                         -1,
                         &stmt,
                         nullptr);
  ASSERT_EQ(r, SQLITE_OK);
  sqlite3_step(stmt);

  ASSERT_EQ(sqlite3_column_int(stmt, 0), 1);
  ASSERT_EQ(sqlite3_column_int(stmt, 1), 1);
  const float *param = (const float *) sqlite3_column_blob(stmt, 2);
  ASSERT_FLOAT_EQ(param[0], 1.0);
  sqlite3_finalize(stmt);

  system("rm database.db");
}

TEST(Image, LoadImage) {
  std::shared_ptr<image> img = std::make_shared<image>();
  bool rc = img->load_image("../unittest/Img/100_7100.JPG");
  ASSERT_TRUE(rc);
  ASSERT_GT(img->get_features_num(), 0);
}

#endif //SFM_SRC_UNITTEST_HPP_
