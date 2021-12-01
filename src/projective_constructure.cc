#include "projective_constructure.hpp"

#include "internal/function_programming.hpp"
#include "iostream"
#include "solver/triangular_solver.hpp"
int64_t Hash(int32_t image_idx, int32_t feature_idx) {
  return ((int64_t)image_idx << 32) + (int64_t)(feature_idx);
}

void UnHash(int64_t hash_value, int32_t& image_idx, int32_t& feature_idx) {
  feature_idx = hash_value & 0xffffffff;
  image_idx = (hash_value >> 32) & 0xffffffff;
}

void Track::InsertHashCode(int64_t hash_code, Observation ob) {
  int32_t image_id, feature_idx;
  UnHash(hash_code, image_id, feature_idx);
  obs[image_id] = ob;
}

void TrackBuilder::InsertMatch(Pair pair, Matches connected_matches) {
  for (Match m : connected_matches) {
    auto lhs_hash = Hash(pair.first, m.lhs_idx);
    auto rhs_hash = Hash(pair.second, m.rhs_idx);
    // there are three case:
    // a. both lhs and rhs features not exists, then we create a new track
    // b. either lhs or rhs exists, then we add the one doesn't exists into the
    // track. c. both lhs and rhs exists, we only process the case that lhs and
    // rhs belong to different track, thus we have to merge them.
    //

    bool lhs_exist = FeatureExists(lhs_hash);
    bool rhs_exist = FeatureExists(rhs_hash);

    if (lhs_exist && rhs_exist) {
      // c.
      auto lhs_track_id = GetTrackId(lhs_hash);
      auto rhs_track_id = GetTrackId(rhs_hash);
      if (lhs_track_id != rhs_track_id) {
        // we merge rhs to lhs simply
        MergeTrack(rhs_track_id, lhs_track_id);
      }
    } else if (!lhs_exist && !rhs_exist) {
      // a.
      CreateNewTrack(lhs_hash, m.lhs_observation, rhs_hash, m.rhs_observation);
    } else {
      // b.
      if (lhs_exist) {
        auto track_id = GetTrackId(lhs_hash);
        AppendFeatureToTrack(track_id, rhs_hash, m.rhs_observation);
      } else {
        auto track_id = GetTrackId(rhs_hash);
        AppendFeatureToTrack(track_id, lhs_hash, m.lhs_observation);
      }
    }
  }
}

bool TrackBuilder::FeatureExists(IndexT image_idx, IndexT feature_idx) const {
  auto hash_code = Hash(image_idx, feature_idx);
  return FeatureExists(hash_code);
}

bool TrackBuilder::FeatureExists(int64_t hash_code) const {
  return hash_to_track_id.find(hash_code) != hash_to_track_id.cend();
}

size_t TrackBuilder::GetTrackId(int64_t hash_code) const {
  return hash_to_track_id.at(hash_code);
}

void TrackBuilder::CreateNewTrack(int64_t hash_code_1, Observation ob1,
                                  int64_t hash_code_2, Observation ob2) {
  size_t track_id = tracks_.size();
  Track t;
  t.InsertHashCode(hash_code_1, ob1);
  t.InsertHashCode(hash_code_2, ob2);
  tracks_.push_back(std::move(t));
  tracks_valid_.push_back(true);
  hash_to_track_id[hash_code_1] = track_id;
  hash_to_track_id[hash_code_2] = track_id;
}

void TrackBuilder::MergeTrack(int64_t from, int64_t to) {
  tracks_[hash_to_track_id[to]] += tracks_[hash_to_track_id[from]];
  // we should tag the track_id of from invalidate.
  tracks_valid_[hash_to_track_id[from]] = false;
  hash_to_track_id[from] = hash_to_track_id[to];
}

void TrackBuilder::AppendFeatureToTrack(size_t track_id, int64_t hash_code,
                                        Observation ob) {
  hash_to_track_id[hash_code] = track_id;
  tracks_[track_id].InsertHashCode(hash_code, ob);
}


    Track& TrackBuilder::GetTrackById(size_t track_id) {
      if (tracks_valid_[track_id] == false) {
        throw std::out_of_range("track id is invalid");
      }
      return tracks_[track_id];
    }

std::vector<Track*> TrackBuilder::AllTrack() { 
  std::vector<Track*> res;
  for (size_t track_id = 0; track_id < tracks_.size(); track_id) {
    if (tracks_valid_[track_id]) {
      res.push_back(&tracks_[track_id]);
    }
  }
  return res;
}

Mat34 ProjectiveStructure::PMatrix(IndexT image_id) const {
  return K_matrix * extrinsic_parameters.at(image_id).P34();
}

ProjectiveStructure::ProjectiveStructure(SfMData& sfm_data)
    : image_ids{},
      remainer_ids{},
      extrinsic_parameters{},
      K_matrix{Mat33::Identity()},
      track_builder{},
      all_matches(sfm_data.matches) {
  for (auto&& [image_ids, _] : sfm_data.views) {
    remainer_ids.insert(image_ids);
  }
}

void ProjectiveStructure::InitializeStructure(const Pair& initial_pair,
                                              Matches& inlier_matches,
                                              Mat34& P1, Mat34& P2) {
  image_ids.insert(initial_pair.first);
  image_ids.insert(initial_pair.second);
  remainer_ids.erase(initial_pair.first);
  remainer_ids.erase(initial_pair.second);

  K_matrix = Eigen::Matrix3d::Identity();

  extrinsic_parameters.insert({initial_pair.first, P1});
  extrinsic_parameters.insert({initial_pair.second, P2});
  track_builder.InsertMatch(initial_pair, inlier_matches);
  // Get the track between initial pair.
  // And Triangular the track.
  // Add a method to Triangular the track.
  // Add a method to refind the track.

  std::vector<Track*> tracks = track_builder.AllTrack();
  // TODO (junlinp@qq.com):
  // there should be some tracks
  // we should triangular all of them.
  // BUG (junlinp@qq.com)
  // DLT result all is (0, 0, 0, 1)^T
  std::printf("Triangluar Structure %lu Points\n", tracks.size());
  for (Track* track_ptr : tracks) {
    // DLT
    std::vector<Mat34> P_matrixs;
    std::vector<Eigen::Vector3d> obs;
    for (auto&& [image_id, ob] : track_ptr->obs) {
      P_matrixs.push_back(PMatrix(image_id));
      obs.push_back(ob.homogeneous());
    }
    Eigen::Vector4d X;
    DLT(P_matrixs, obs, X);
    track_ptr->X = X.hnormalized();
  }
}

Correspondence ProjectiveStructure::FindCorrespondence(IndexT image_id) const {
  Correspondence correspondence;
  for (IndexT constructed_id : image_ids) {
    Matches matches = GetMatches(constructed_id, image_id);
    //std::printf("%lld - %lld with %lu matches\n", constructed_id, image_id,
    // matches.size());
    auto functor = [this, image_id_ = constructed_id](const Match& m) -> bool {
      return this->track_builder.FeatureExists(image_id_, m.lhs_idx);
    };
    // The Function programing has bug
    //temp_matches = matches | Filter(functor) | ToVector();
    Matches temp_matches;
    for (Match m : matches) {
      if (track_builder.FeatureExists(constructed_id, m.lhs_idx)) {
        temp_matches.push_back(m);
      }
    }
    temp_matches = matches | Filter(functor) | ToVector();
    //std::printf("Matches %lu\n", temp_matches.size());
    if (!temp_matches.empty()) {
      for (Match m : temp_matches) {
        auto lhs_hash = Hash(constructed_id, m.lhs_idx);

        auto track_id = track_builder.GetTrackId(lhs_hash);
        auto track = const_cast<ProjectiveStructure*>(this)->track_builder.GetTrackById(track_id);
        correspondence.Insert(&track, m);
        //correspondence.Insert(track_id, m);
      }
    }
  }
  return correspondence;
}

IndexT ProjectiveStructure::NextImage() const {
  int max_correspondence_num = 0;
  IndexT best_idx = -1;
  for (IndexT remainer_id : remainer_ids) {
    auto correspondence = FindCorrespondence(remainer_id);
    if (correspondence.size() > max_correspondence_num) {
      max_correspondence_num = correspondence.size();
      best_idx = remainer_id;
    }
  }
  return best_idx;
}
void ProjectiveStructure::UnRegister(IndexT image_id) {
  remainer_ids.erase(image_id);
}

void ProjectiveStructure::Register(IndexT image_id, Mat34 P, Correspondence correspondence) {
  // TODO: add correspondence into track
  UnRegister(image_id);
  image_ids.insert(image_id);
  extrinsic_parameters.insert({image_id, P});
}

void ProjectiveStructure::TriangularNewPoint(IndexT image_id) {
  // TODO (junlinp@qq.com) :
  // Implement
}
void ProjectiveStructure::LocalBundleAdjustment() {
  // TODO (junlinp@qq.com) :
  // Implement
}

Matches ProjectiveStructure::GetMatches(IndexT constructed_id,
                                        IndexT need_to_register_id) const {
  Matches matches;

  if (all_matches.find({constructed_id, need_to_register_id}) !=
      all_matches.end()) {
    return all_matches.at({constructed_id, need_to_register_id});
  }

  if (all_matches.find({need_to_register_id, constructed_id}) !=
      all_matches.end()) {
    Matches temp = all_matches.at({need_to_register_id, constructed_id});
    for (Match m : temp) {
      std::swap(m.lhs_observation, m.rhs_observation);
      std::swap(m.lhs_idx, m.rhs_idx);
      matches.push_back(m);
    }
  }
  return matches;
}