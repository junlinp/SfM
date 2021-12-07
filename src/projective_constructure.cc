#include "projective_constructure.hpp"

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
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

  if (obs.find(image_id) != obs.end()) {
    std::printf("Warning: Insert image_id into same track twice\n");
  }
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

size_t TrackBuilder::CreateNewTrack(int64_t hash_code_1, Observation ob1,
                                    int64_t hash_code_2, Observation ob2) {
  size_t track_id = tracks_.size();
  Track t;
  t.InsertHashCode(hash_code_1, ob1);
  t.InsertHashCode(hash_code_2, ob2);
  tracks_.push_back(std::move(t));
  tracks_valid_.push_back(true);
  hash_to_track_id[hash_code_1] = track_id;
  hash_to_track_id[hash_code_2] = track_id;
  return track_id;
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

void TrackBuilder::AppendCorrespondence(IndexT image_id, Correspondence cor) {
  for (auto&& [track_ptr, match] : cor.cor_) {
    track_ptr->InsertHashCode(Hash(image_id, match.rhs_idx),
                              match.rhs_observation);
  }
}

Track& TrackBuilder::GetTrackById(size_t track_id) {
  if (tracks_valid_[track_id] == false) {
    throw std::out_of_range("track id is invalid");
  }
  return tracks_[track_id];
}

std::vector<Track*> TrackBuilder::AllTrack() {
  std::vector<Track*> res;
  for (size_t track_id = 0; track_id < tracks_.size(); track_id++) {
    if (tracks_valid_[track_id]) {
      res.push_back(&tracks_[track_id]);
    }
  }
  return res;
}

Mat34 ProjectiveStructure::PMatrix(IndexT image_id) const {
  return extrinsic_parameters.at(image_id);
}

ProjectiveStructure::ProjectiveStructure(SfMData& sfm_data)
    : image_ids{},
      remainer_ids{},
      extrinsic_parameters{},
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

  extrinsic_parameters.insert({initial_pair.first, P1});
  extrinsic_parameters.insert({initial_pair.second, P2});
  track_builder.InsertMatch(initial_pair, inlier_matches);
  // Get the track between initial pair.
  // And Triangular the track.
  // Add a method to Triangular the track.
  // Add a method to refind the track.

  std::vector<Track*> tracks = track_builder.AllTrack();

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
    // BundleAdjustmentTriangular(P_matrixs, obs, X);
    DLT(P_matrixs, obs, X);
    track_ptr->X = X.hnormalized();
  }
}

Correspondence ProjectiveStructure::FindCorrespondence(IndexT image_id) const {
  Correspondence correspondence;
  for (IndexT constructed_id : image_ids) {
    Matches matches = GetMatches(constructed_id, image_id);
    // std::printf("%lld - %lld with %lu matches\n", constructed_id, image_id,
    //  matches.size());
    auto functor = [this, image_id_ = constructed_id](const Match& m) -> bool {
      return this->track_builder.FeatureExists(image_id_, m.lhs_idx);
    };
    // The Function programing has bug
    // temp_matches = matches | Filter(functor) | ToVector();
    Matches temp_matches;
    for (Match m : matches) {
      if (track_builder.FeatureExists(constructed_id, m.lhs_idx)) {
        temp_matches.push_back(m);
      }
    }
    temp_matches = matches | Filter(functor) | ToVector();
    // std::printf("Matches %lu\n", temp_matches.size());
    if (!temp_matches.empty()) {
      for (Match m : temp_matches) {
        auto lhs_hash = Hash(constructed_id, m.lhs_idx);

        auto track_id = track_builder.GetTrackId(lhs_hash);
        // std::printf("Track Id %lu\n", track_id);
        auto&& track =
            const_cast<ProjectiveStructure*>(this)->track_builder.GetTrackById(
                track_id);
        // std::printf("Ptr %p\n", &track);
        correspondence.Insert(&track, m);
        // correspondence.Insert(track_id, m);
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

void ProjectiveStructure::Register(IndexT image_id, Mat34 P,
                                   Correspondence correspondence) {
  // TODO: add correspondence into track
  UnRegister(image_id);
  image_ids.insert(image_id);
  extrinsic_parameters.insert({image_id, P});

  // add correspondence to track
  track_builder.AppendCorrespondence(image_id, correspondence);
}

namespace {
double Error(const Mat34& P1, const Observation& ob1, Eigen::Vector4d& X) {
  Eigen::Vector3d uv = P1 * X;
  return (ob1 - uv.hnormalized()).squaredNorm();
}

double Error(const Mat34& P1, const Observation& ob1, const Mat34& P2,
             const Observation& ob2, Eigen::Vector4d& X) {
  return (Error(P1, ob1, X) + Error(P2, ob2, X)) / 2.0;
}
}  // namespace
void ProjectiveStructure::TriangularNewPoint(IndexT image_id) {
  // TODO (junlinp@qq.com) :
  // Implement

  // find some match have not been a track between image_id and reconstracted_id
  std::size_t add_track_count = 0;
  for (auto constructed_id : image_ids) {
    if (constructed_id == image_id) continue;
    Matches matches = GetMatches(constructed_id, image_id);
    for (Match match : matches) {
      auto lhs_hash = Hash(constructed_id, match.lhs_idx);
      auto rhs_hash = Hash(image_id, match.rhs_idx);

      auto lhs_exist = track_builder.FeatureExists(lhs_hash);
      auto rhs_exist = track_builder.FeatureExists(rhs_hash);
      if (!lhs_exist && !rhs_exist) {
        std::vector<Mat34> P;
        std::vector<Observation> ob_vec;
        P.push_back(extrinsic_parameters.at(constructed_id));
        P.push_back(extrinsic_parameters.at(image_id));
        ob_vec.push_back(match.lhs_observation);
        ob_vec.push_back(match.rhs_observation);

        Eigen::Vector4d X;
        DLT(P, ob_vec | Transform([](Observation ob) -> Eigen::Vector3d {
                 return ob.homogeneous();
               }) | ToVector(),
            X);
        if (Error(P[0], ob_vec[0], P[1], ob_vec[1], X) < 3.84) {
          auto track_id = track_builder.CreateNewTrack(
              lhs_hash, match.lhs_observation, rhs_hash, match.rhs_observation);
          Track& track = track_builder.GetTrackById(track_id);
          track.X = X.hnormalized();
          add_track_count++;
        }
      }
    }
  }

  std::printf("Triangular %lu new Points\n", add_track_count);
}

struct ReProjectiveError {
  ReProjectiveError(const Observation ob) : ob_{ob} {}
  Observation ob_;
  template <typename T>
  bool operator()(const T* input_P, const T* input_X, T* output) const {
    T uv[3];

    for (size_t row = 0; row < 3; row++) {
      uv[row] = input_P[row * 4 + 3];
      for (size_t col = 0; col < 3; col++) {
        uv[row] += input_P[row * 4 + col] * input_X[col];
      }
    }

    uv[0] /= uv[2];
    uv[1] /= uv[2];

    output[0] = uv[0] - ob_.x();
    output[1] = uv[1] - ob_.y();
    return true;
  }
};

double LocalProjectiveBundleAdjustment(
    std::map<IndexT, Mat34>& extrinsic_parameters,
    std::vector<Track*>& tracks) {
  ceres::Problem problem;

  for (Track* track_ptr : tracks) {
    for (auto&& [image_id, observation] : track_ptr->obs) {
      Mat34& P = extrinsic_parameters[image_id];
      Eigen::Vector3d& X = track_ptr->X;

      ceres::CostFunction* cost_function_ptr =
          new ceres::AutoDiffCostFunction<ReProjectiveError, 2, 3 * 4, 3>(
              new ReProjectiveError(observation));
      problem.AddResidualBlock(cost_function_ptr, nullptr, P.data(), X.data());
    }
  }

  ceres::Solver::Summary sumary;
  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 500;

  ceres::Solve(solver_options, &problem, &sumary);

  if (sumary.IsSolutionUsable()) {
    std::cout << sumary.BriefReport() << std::endl;
    return std::sqrt(sumary.final_cost / sumary.num_residual_blocks / 2.0);
  } else {
    return std::numeric_limits<double>::lowest();
  }
}
void ProjectiveStructure::LocalBundleAdjustment() {
  // Each Track build a Error Function

  // TODO (junlinp@qq.com):
  // Reserve the Identity Camera Matrix.
  //
  std::vector<Track*> tracks = track_builder.AllTrack();

  double final_objective_cost =
      LocalProjectiveBundleAdjustment(extrinsic_parameters, tracks);
  std::cout << "Local RMSE : " << final_objective_cost << std::endl;
}

std::vector<Track> ProjectiveStructure::GetTracks() const {
  std::vector<Track*> tracks =
      const_cast<ProjectiveStructure*>(this)->track_builder.AllTrack();
  std::vector<Track> res =
      tracks | Transform([](Track* ptr) -> Track { return *ptr; }) | ToVector();
  return res;
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