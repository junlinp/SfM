#include "euclidean_constructure.hpp"

#include "solver/algebra.hpp"
#include "solver/self_calibration_solver.hpp"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "internal/function_programming.hpp"
#include "iostream"
#include "solver/triangular_solver.hpp"
#include "solver/resection_solver.hpp"
#include "solver/fundamental_solver.hpp"
#include "ceres/problem.h"
#include "ceres/autodiff_cost_function.h"
#include "solver/bundle_adjustment.hpp"
#include "ceres/solver.h"
#include "ceres/rotation.h"
#include "ceres/loss_function.h"
#include "solver/epnp_solver.hpp"

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

void TrackBuilder::InsertTriple(TripleIndex triple_index, TripleMatch match, Eigen::Vector3d X) {
  auto I_hash = Hash(triple_index.I, match.I_idx);
  auto J_hash = Hash(triple_index.J, match.J_idx);
  auto K_hash = Hash(triple_index.K, match.K_idx);

  size_t track_id = CreateNewTrack(I_hash, match.I_observation, J_hash, match.J_observation);
  AppendFeatureToTrack(track_id, K_hash, match.K_observation);
  Track& track = GetTrackById(track_id);
  track.X = X;
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

EuclideanStructure::EuclideanStructure(const SfMData& sfm_data)
    : image_ids{},
      remainer_ids{},
      track_builder{},
      all_matches(sfm_data.matches),
      sfm_data_{sfm_data} {
  for (auto&& [image_ids, _] : sfm_data.views) {
    remainer_ids.insert(image_ids);
  }
}

struct MatchesWrap {
  std::map<Pair, Matches> matches;
  MatchesWrap(std::map<Pair, Matches> matches) : matches(matches) {}

  bool Exists(Pair pair) const {
    auto origin_iterator = matches.find(pair);
    auto reverse_iterator = matches.find({pair.second, pair.first});

    return origin_iterator != matches.end() || reverse_iterator != matches.end();
  }

  Matches at(Pair pair) const {

    if (matches.find(pair) != matches.end()) {
      return matches.at(pair);
    }

    if (matches.find({pair.second, pair.first}) != matches.end()) {
      Matches res;
      const Matches& reverse = matches.at({pair.second, pair.first});
      for (Match match : reverse) {
        std::swap(match.lhs_observation, match.rhs_observation);
        std::swap(match.lhs_idx, match.rhs_idx);
        res.push_back(match);
      }
      return res;
    }

    return Matches{};
  }
};
std::vector<TripleIndex> ListTriple(const SfMData& sfm_data) {
  std::vector<TripleIndex> res;
  std::vector<IndexT> total_index = sfm_data.views | Transform([](auto pair){return pair.first;}) | ToVector();
  size_t max_size = total_index.size();

  for(size_t i = 0; i < max_size; i++) {
    for (size_t j = i + 1; j < max_size; j++) {
      for (size_t k = j + 1; k < max_size; k++) {
        res.push_back(TripleIndex{total_index[i], total_index[j], total_index[k]});
      }
    }
  }
  return res;
}

bool ValidateTriple(TripleIndex index,const MatchesWrap& match) {
  return match.Exists({index.I, index.J}) && match.Exists({index.J, index.K}) && match.Exists({index.K, index.I});
}

std::vector<TripleMatch> ConstructTripleMatch(TripleIndex index, const MatchesWrap& match) {

  Matches I_J_matches = match.at({index.I, index.J});
  Matches J_K_matches = match.at({index.J, index.K});
  Matches K_I_matches = match.at({index.K, index.I});
  
  std::map<IndexT, Match> j_match_k;
  std::map<IndexT, Match> k_match_i;
  for (Match j_k : J_K_matches) {
    j_match_k[j_k.lhs_idx] = j_k;
  }

  for (Match k_i : K_I_matches) {
    k_match_i[k_i.lhs_idx] = k_i;
  }
  std::vector<TripleMatch> res;

  for (Match i_j : I_J_matches) {
    // J should has match to k
    IndexT j_index = i_j.rhs_idx;
    if (j_match_k.find(j_index) != j_match_k.end()) {
      Match j_k = j_match_k[j_index];
      IndexT k_index = j_k.rhs_idx;

      if (k_match_i.find(k_index) != k_match_i.end()) {
        Match k_i = k_match_i.at(k_index);
        IndexT i_index = k_i.rhs_idx;

        if (i_j.lhs_idx == i_index) {
          TripleMatch m;
          m.I_idx = i_index; m.J_idx = j_index; m.K_idx = k_index;
          m.I_observation = i_j.lhs_observation;
          m.J_observation = i_j.rhs_observation;
          m.K_observation = j_k.rhs_observation;
          res.push_back(m);
        }
      }
    }



  }
  return res;
}
std::pair<TripleIndex, std::vector<TripleMatch>> FindInitialTriple(const SfMData& sfm_data) {
  
  MatchesWrap match(sfm_data.matches);

  auto temp = ListTriple(sfm_data) | Transform([match](auto triple_index) { return std::pair<TripleIndex, std::vector<TripleMatch>>{triple_index, ConstructTripleMatch(triple_index, match)};}) | ToVector();
  
  std::sort(temp.begin(), temp.end(),[](auto lhs_pair, auto rhs_pair) { return lhs_pair.second.size() > rhs_pair.second.size();});
  return temp[0];
}


std::ostream& operator<<(std::ostream& os, TripleIndex index) {
  os << "[" << index.I << "," << index.J << "," << index.K << "]";
  return os;
}

std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>
Convert(const Matches& matches) {

 using T = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

 std::vector<T> res;
 for (Match m : matches) {
   T t{m.lhs_observation, m.rhs_observation};

   res.push_back(t);
 }
 return res;
}

Pose RecoveryRC(Mat34 P, Mat33 K) {
  P = K.inverse() * P;
  Mat33 beta_R =  P.block(0, 0, 3, 3);
  Eigen::Vector3d beta_t = P.block(0, 3, 3, 1);
  auto svd = beta_R.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat33 U = svd.matrixU();
  Mat33 V = svd.matrixV();
  Eigen::Vector3d S = svd.singularValues();

  double beta = S.array().mean();
  Mat33 R = U*V.transpose();
  Eigen::Vector3d C = - R.transpose() * beta_t / beta;
  return Pose{R, C};
}
auto EuclideanStructure::BundleAdjustmentProblem() {
  ceres::Problem problem;

  std::vector<Track*> all_track = track_builder.AllTrack();

  for (Track* track_ptr : all_track) {
    for (auto ob_iter : track_ptr->obs) {
      int camera_index = ob_iter.first;
      Observation ob = ob_iter.second;
      Pose& pose = poses[camera_index];
      Eigen::Vector3d& X = track_ptr->X;
      // cost_function
      if (camera_index == const_camera_matrix_index) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ConstRCCostFunctor, 2, 4, 3>(
                new ConstRCCostFunctor(pose.angle_axis(), pose.center(),
                                       ob.data()));
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(4.0),
                                 K_.data(), X.data());
      } else {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<FreeCostFunctor, 2, 4, 3, 3, 3>(
                new FreeCostFunctor(ob.data()));
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(4.0), K_.data(), pose.angle_axis(), pose.center(), X.data());
      }
    }
  }
  return problem;
}

void EuclideanStructure::StartReconstruction() {
    InitializeStructure();
    IncrementalRegister();
}

void EuclideanStructure::InitializeStructure() {
  auto&& [triple_index, initial_triple_match] = FindInitialTriple(sfm_data_);
  std::cout << "Choose "<< triple_index << " with " << initial_triple_match.size() << " Matches To Initialize"<< std::endl; 

  Matches two_match;

  std::vector<Eigen::Vector2d> obs;
  for (TripleMatch match : initial_triple_match) {
    Match m;
    m.lhs_idx = match.I_idx;
    m.lhs_observation = match.I_observation;
    m.rhs_idx = match.J_idx;
    m.rhs_observation = match.J_observation;
    two_match.push_back(m);

    obs.push_back(match.K_observation);
  }


  EightPointFundamentalSolver fundamental_solver;
  Mat33 F;
  fundamental_solver.Fit(Convert(two_match), F);
  
  Eigen::Vector3d e1 = NullSpace(F.transpose());

  Mat34 P1, P2;
  P1 << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0 ,0.0, 0.0,
        0.0, 0.0, 1.0, 0.0;
  P2 << SkewMatrix(e1) * F, e1;


  std::vector<Eigen::Vector3d> points;
  for (Match m : two_match) {
    Eigen::Vector4d X;
    DLT({P1, P2}, {m.lhs_observation.homogeneous(), m.rhs_observation.homogeneous()}, X);
    points.push_back(X.hnormalized());
  }
  
  Mat34 P3;
  DLTSolver resection_solver;
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector3d>> data_points;
  for(int i = 0; i < points.size(); i++) {
    data_points.push_back({obs[i], points[i]});
  }
  resection_solver.Fit(data_points, &P3);
  std::cout << "P3 : " << std::endl << P3 << std::endl;
  ceres::Problem problem;
  for (int i = 0; i < points.size(); i++) {
    ceres::CostFunction* cost_P1 = new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(new ConstCameraMatrixCostFunctor(P1, initial_triple_match[i].I_observation)) ;
    problem.AddResidualBlock(cost_P1, nullptr, points[i].data());

    ceres::CostFunction* cost_p2 = new ceres::AutoDiffCostFunction<CostFunctor,2, 12, 3>(new CostFunctor(initial_triple_match[i].J_observation));
    problem.AddResidualBlock(cost_p2, nullptr, P2.data(), points[i].data());

    ceres::CostFunction* cost_p3 = new ceres::AutoDiffCostFunction<CostFunctor,2, 12, 3>(new CostFunctor(initial_triple_match[i].K_observation));
    problem.AddResidualBlock(cost_p3, nullptr, P3.data(), points[i].data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 1024;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.IsSolutionUsable() << std::endl;

  std::cout << summary.BriefReport() << std::endl;

  std::cout << "P1 : " << std::endl << P1 << std::endl;
  std::cout << "P2 : " << std::endl << P2 << std::endl;
  std::cout << "P3 : " << std::endl << P3 << std::endl;

  IterativeSolver self_calibration_solver(sfm_data_.image_width, sfm_data_.image_height, sfm_data_.image_width / 2.0, sfm_data_.image_height / 2.0);
  self_calibration_solver.Solve(std::vector<Mat34>{P1, P2, P3});

  Eigen::Matrix3d K = self_calibration_solver.K();

  std::cout << "Calibrated Camera Matrix : " << std::endl << K << std::endl;


  // Initial Reconstruction
  std::cout << "P1 : " << P1 * self_calibration_solver.HomogeneousMatrix() << std::endl;
  std::cout << "P2 : " << P2 * self_calibration_solver.HomogeneousMatrix() << std::endl;
  std::cout << "P3 : " << P3 * self_calibration_solver.HomogeneousMatrix() << std::endl;
  
  for (Eigen::Vector3d& x : points) {
    x = (self_calibration_solver.HomogeneousMatrix().inverse() * x.homogeneous()).hnormalized();
  }
  P1 = P1 * self_calibration_solver.HomogeneousMatrix();
  P2 = P2 * self_calibration_solver.HomogeneousMatrix();
  P3 = P3 * self_calibration_solver.HomogeneousMatrix();
  K_ = K;

  const_camera_matrix_index = triple_index.I;

  poses[triple_index.I] = RecoveryRC(P1, K);
  poses[triple_index.J] = RecoveryRC(P2, K);
  poses[triple_index.K] = RecoveryRC(P3, K);

  for (int i = 0; i < points.size(); i++) {
    track_builder.InsertTriple(triple_index, initial_triple_match[i], points[i]);
  }

  ceres::Problem p = BundleAdjustmentProblem();
  options.max_num_iterations = 1024;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.IsSolutionUsable() << std::endl;
  std::cout << "Local Refine Bundle Adjustment : " << std::endl << summary.BriefReport() << std::endl;
  std::cout << "K Matrix : " << K_.K() << std::endl;

  image_ids.insert(triple_index.I);
  image_ids.insert(triple_index.J);
  image_ids.insert(triple_index.K);

  remainer_ids.erase(triple_index.I);
  remainer_ids.erase(triple_index.J);
  remainer_ids.erase(triple_index.K);
}

void EuclideanStructure::IncrementalRegister() {
   IndexT next_index = -1;
   while( (next_index = NextImage()) != -1) {
     std::cout << "Register : " << next_index << std::endl;
     Correspondence correspondence = FindCorrespondence(next_index);
     auto data_points = correspondence.Get2D_3D();
     std::cout << "Try to Estimate Rotation and Center with " << data_points.size() << " Points"  << std::endl;
     // TODO: should using ransac
     // use 
     Pose P;
     solver::EPnPSolver solver{K_.K()};
     solver.Fit(data_points, &P);
     Register(next_index, P, correspondence);
     
     LocalBundleAdjustment();
     TriangularNewPoint(next_index);
     LocalBundleAdjustment();
   }
}

Correspondence EuclideanStructure::FindCorrespondence(IndexT image_id) const {
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
            const_cast<EuclideanStructure*>(this)->track_builder.GetTrackById(
                track_id);
        // std::printf("Ptr %p\n", &track);
        correspondence.Insert(&track, m);
        // correspondence.Insert(track_id, m);
      }
    }
  }
  return correspondence;
}

IndexT EuclideanStructure::NextImage() const {
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

//void EuclideanStructure::UnRegister(IndexT image_id) {
//  remainer_ids.erase(image_id);
//}

void EuclideanStructure::Register(IndexT image_id, Pose P,
                                   Correspondence correspondence) {
  // TODO: add correspondence into track
  remainer_ids.erase(image_id);
  image_ids.insert(image_id);
  poses[image_id] = P;
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
  return (Error(P1, ob1, X) + Error(P2, ob2, X));
}
}  // namespace
void EuclideanStructure::TriangularNewPoint(IndexT image_id) {
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
        ob_vec.push_back(match.lhs_observation);
        ob_vec.push_back(match.rhs_observation);
        Pose constructed_id_pose = poses[constructed_id];
        Pose image_id_pose = poses[image_id];
        P.push_back(K_.K() * constructed_id_pose.P());
        P.push_back(K_.K() * image_id_pose.P());
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
    std::vector<Track*>& tracks,
    IndexT const_camera_matrix_index
    ) {
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
  Mat34& const_P = extrinsic_parameters[const_camera_matrix_index];
  problem.SetParameterBlockConstant(const_P.data());

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
void EuclideanStructure::LocalBundleAdjustment() {
  ceres::Problem problem = BundleAdjustmentProblem();

  ceres::Solver::Options options;
  options.max_num_iterations = 1024;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Local Bundle Adjustment" << std::endl 
  << summary.BriefReport() << std::endl;
}

std::vector<Track> EuclideanStructure::GetTracks() const {
  std::vector<Track*> tracks =
      const_cast<EuclideanStructure*>(this)->track_builder.AllTrack();
  std::vector<Track> res =
      tracks | Transform([](Track* ptr) -> Track { return *ptr; }) | ToVector();
  return res;
}

Matches EuclideanStructure::GetMatches(IndexT constructed_id,
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