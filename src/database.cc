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
                    "id integer AUTO_INCREMENT,"
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
        "id integer AUTO_INCREMENT,"
        "qw double,"
        "qx double,"
        "qy double,"
        "qz double,"
        "tx double,"
        "ty double,"
        "tz double,"
        "camera_id integer,"
        "name varchar(128),"
        "PRIMARY KEY(id)"
        ")";
  rc = sqlite3_exec(sqlite_handle, sql, NULL, NULL, &errmsg);
  if (rc) {
    printf("Error : %s\n", errmsg);
    return false;
  }
  sql = "CREATE TABLE t_keypoint("
        "id integer AUTO_INCREMENT,"
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
        "id integer AUTO_INCREMENT,"
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
