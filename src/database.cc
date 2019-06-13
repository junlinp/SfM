//
// Created by junlinp on 2019-06-09.
//

#include "database.hpp"
database::database(std::string name) {
  this->name = name;
  int rc = sqlite3_open(name.c_str(), &this->sqlite_handle);
  if (rc) {
    // error
  }
}
bool database::create() {
  const char* sql = "CREATE TABLE t_camera("
                    "id INTEGER NOT NULL,"
                    "camera_model integer,"
                    "camera_param_num integer,"
                    "camera_param blob,"
                    "PRIMARY KEY(id)"
                    ")";
  char* errmsg = nullptr;
  int rc = sqlite3_exec(sqlite_handle, sql, NULL, NULL, &errmsg);
  if (rc) {
    printf("Error : %s\n", errmsg);
    return false;
  }
  sql = "CREATE TABLE t_image("
        "id INTEGER NOT NULL,"
        "qw double,"
        "qx double,"
        "qy double,"
        "qz double,"
        "tx double,"
        "ty double,"
        "tz double,"
        "camera_id integer,"
        "name varchar(128),"
        "width integer,"
        "height integer,"
        "PRIMARY KEY(id)"
        ")";
  rc = sqlite3_exec(sqlite_handle, sql, NULL, NULL, &errmsg);
  if (rc) {
    printf("Error : %s\n", errmsg);
    return false;
  }
  sql = "CREATE TABLE t_keypoint("
        "id INTEGER NOT NULL,"
        "image_id integer,"
        "idx integer,"
        "x double,"
        "y double,"
        "PRIMARY KEY(id)"
        ")";
  rc = sqlite3_exec(sqlite_handle, sql, NULL, NULL, &errmsg);
  if (rc) {
    printf("Error : %s\n", errmsg);
    return false;
  }
  sql = "CREATE TABLE t_descript("
        "id INTEGER NOT NULL,"
        "image_id integer,"
        "idx integer,"
        "width integer,"
        "bytes blob,"
        "PRIMARY KEY(id)"
        ")";
  rc = sqlite3_exec(sqlite_handle, sql, NULL, NULL, &errmsg);
  if (rc) {
    printf("Error : %s\n", errmsg);
    return false;
  }

  return true;
}
database::~database() {
  sqlite3_close(this->sqlite_handle);
}
std::shared_ptr<image> database::GetImageById(int id) {

  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(sqlite_handle, "SELECT qw,qx,qy,qz,tx,ty,tz,camera_id, name, width, height from t_image where id = ?", -1, &stmt, nullptr);

  if (rc != SQLITE_OK) {
    fprintf(stderr, "%s\n", sqlite3_errmsg(sqlite_handle));
    return std::shared_ptr<image>(nullptr);
  }

  rc = sqlite3_bind_int(stmt, 1, id);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "%s\n", sqlite3_errmsg(sqlite_handle));
    return std::shared_ptr<image>(nullptr);
  }

  rc = sqlite3_step(stmt);

  if (rc != SQLITE_ROW) {
    fprintf(stderr, "%s\n", sqlite3_errmsg(sqlite_handle));
    return std::shared_ptr<image>(nullptr);

  }

  auto img = std::make_shared<image>();

  img->GetQvec()[0] = sqlite3_column_double(stmt, 0);
  img->GetQvec()[1] = sqlite3_column_double(stmt, 1);
  img->GetQvec()[2] = sqlite3_column_double(stmt, 2);
  img->GetQvec()[3] = sqlite3_column_double(stmt, 3);

  img->GetTvec()[0] = sqlite3_column_double(stmt, 4);
  img->GetTvec()[1] = sqlite3_column_double(stmt, 5);
  img->GetTvec()[2] = sqlite3_column_double(stmt, 6);

  img->set_camera_id( sqlite3_column_int(stmt, 7));
  img->SetName( std::string( (const char*)(sqlite3_column_text(stmt, 8))));

  img->SetWidth(sqlite3_column_int(stmt, 9));
  img->SetHeight(sqlite3_column_int(stmt, 10));
  sqlite3_finalize(stmt);

  sqlite3_prepare_v2(sqlite_handle, "SELECT x, y from t_keypoint where image_id = ? ORDER BY idx ASC", -1, &stmt, nullptr);
  sqlite3_bind_int(stmt, 1, id);

  while( (rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    auto kp = std::make_shared<keypoint>();
    kp->x = sqlite3_column_int(stmt, 0);
    kp->y = sqlite3_column_int(stmt, 1);
    img->keypoint_push_back(kp);
  }

  if (SQLITE_DONE == rc) {
    sqlite3_finalize(stmt);
  } else {
    fprintf(stderr, "%s\n", sqlite3_errmsg(sqlite_handle));
    return nullptr;
  }

  sqlite3_prepare_v2(sqlite_handle, "SELECT width, bytes from t_descriptor where image_id = ? ORDER BY idx ASC", -1, &stmt, nullptr);
  sqlite3_bind_int(stmt, 1, id);

  while( (rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int width = sqlite3_column_int(stmt, 0);
    const void* bytes = sqlite3_column_blob(stmt, 1);
    auto des = std::make_shared<descriptor>((const float*)bytes, width);
    img->descripotr_push_back(des);
  }
  if (SQLITE_DONE == rc) {
    sqlite3_finalize(stmt);
  } else {
    fprintf(stderr, "%s\n", sqlite3_errmsg(sqlite_handle));
    return nullptr;
  }

  sqlite3_finalize(stmt);
  return img;
}
