#include "sfm_data.hpp"
#include <unordered_set>

struct Pose {
    Pose() =default;
    Pose(const Mat34&) {}

    Mat33 R;
    Eigen::Vector3d C;
    Mat34 P34() const {
      Mat34 p;
      p << R, -R * C;
      return p;
    }
};

class Correspondence {
  public:
  size_t size() { return cor_.size(); }
/*
  void Insert(int64_t track_id, Match& match) {
    cor_.insert({track_id, match});
  };
  */

  void Insert(Track* track_ptr, Match& match) {
    cor_.insert({track_ptr, match});
  }

  std::vector<Observation> Getobservations() {
    //
    std::vector<Observation> obs;
    for (auto&& [_, match] : cor_) {
      obs.push_back(match.rhs_observation);
    }
    return obs;
  }

  std::vector<Eigen::Vector3d> GetPoints() {
    // Need Track to Get the X
    std::vector<Eigen::Vector3d> points;
    for (auto&& [ptr, _] : cor_) {
      points.push_back(ptr->X);
    }
    return points;
  }

  private:
  // Track_id, Match
  // ptr_to_track, Match
  //std::map<IndexT, Match> cor_;
  std::map<Track*, Match> cor_;
  
};

template <typename IndexType>
class UnionFindSet {
 public:
  void insertIdx(IndexType idx) { idx_to_parent[idx] = idx; }

  IndexType FindRoot(IndexType idx) const{
    while (idx_to_parent.at(idx) != idx) {
      idx = idx_to_parent.at(idx);
    }
    return idx;
  }

  void Union(IndexType lhs_idx, IndexType rhs_idx) {
    IndexType lhs_root = FindRoot(lhs_idx);
    IndexType rhs_root = FindRoot(rhs_idx);
    idx_to_parent[lhs_root] = rhs_root;
  }

  IndexType DifferentSetSize() const {
    std::set<IndexType> roots;
    for (auto pair : idx_to_parent) {
      auto root = FindRoot(pair.first);
      roots.insert(root);
      idx_to_parent[pair.first] = root;
    }
    return roots.size();
  }

    bool Exists(IndexType idx) const {
        return idx_to_parent.find(idx) != idx_to_parent.end();
    }
 private:
  std::map<IndexType, IndexType> idx_to_parent;
};

struct Track {
    // <ImageId>
    //std::map<IndexT, IndexT> obs;
    std::map<IndexT, Observation> obs;
    Eigen::Vector3d X;

    void InsertHashCode(int64_t hash_code, Observation ob);

    Track& operator+=(const Track& rhs) {
      for (auto&& item : rhs.obs) {
        obs.insert(item);
      }
      return *this;
    }
};

class TrackBuilder {
    private:
    std::map<int64_t, size_t> hash_to_track_id;
    std::vector<Track> tracks_;
    std::vector<bool> tracks_valid_;
    public:
    void InsertMatch(Pair pair, Matches connected_matches);
    bool FeatureExists(IndexT image_idx, IndexT feature_idx) const;
    bool FeatureExists(int64_t hash_code) const;
    size_t GetTrackId(int64_t hash_code) const;
    void CreateNewTrack(int64_t hash_code_1, Observation ob_1, int64_t hash_code_2, Observation ob_2);
    void MergeTrack(int64_t from, int64_t to);
    void AppendFeatureToTrack(size_t track_id, int64_t hash_code, Observation ob);
    Track& GetTrackById(size_t track_id);
    std::vector<Track*> AllTrack();

};

class ProjectiveStructure {

    std::set<IndexT> image_ids;
    std::set<IndexT> remainer_ids;

    std::map<IndexT, Pose> extrinsic_parameters;
    Eigen::Matrix3d K_matrix;
    TrackBuilder track_builder;

    std::map<Pair, Matches> all_matches;

    Matches GetMatches(IndexT constructed_id, IndexT need_to_register_id) const;

    Mat34 PMatrix(IndexT image_id) const;
public:
    ProjectiveStructure(SfMData& sfm_data);

    void InitializeStructure(const Pair& initial_pair, Matches& inlier_matches, Mat34& P1, Mat34& P2);

    Correspondence FindCorrespondence(IndexT image_id) const; 

    IndexT NextImage() const;
    void UnRegister(IndexT image_id);
    void Register(IndexT image_id, Mat34 P, Correspondence correspondence);
    void TriangularNewPoint(IndexT image_id);
    void LocalBundleAdjustment();
    // UnionFindSet
};