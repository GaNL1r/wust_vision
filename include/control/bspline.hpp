#pragma once
#include <Eigen/Dense>
#include <deque>
#include <iostream>
#include <vector>

class RealtimeBSplineSegment {
public:
  RealtimeBSplineSegment(int degree = 3, double step_size = 0.01)
      : k(degree), delta_u(step_size), u(0.0) {}

  // 设置控制点，并构造全局 knots
  void setControlPoints(const std::vector<Eigen::Vector2d> &new_pts,
                        bool smooth = false) {
    full_points = smooth ? smoothPoints(new_pts) : new_pts;
    current_index = 0;
    u = 0.0;
    generateGlobalKnots();
    updateSegment();
  }

  // 添加新控制点（支持滑窗实时添加）
  void addControlPoint(const Eigen::Vector2d &pt) {
    full_points.push_back(pt);
    generateGlobalKnots();
    if (full_points.size() >= k + 1 && !segment_ready)
      updateSegment();
  }

  bool ready() const { return segment_ready; }

  // 评估下一采样点（根据 delta_u 推进）
  Eigen::Vector2d evaluateNext() {
    if (!segment_ready)
      return Eigen::Vector2d::Zero();

    Eigen::Vector2d pt = deBoorEvaluate(u);
    u += delta_u;

    if (u >= 1.0 - 1e-6) {
      u = 0.0;
      ++current_index;
      updateSegment();
    }

    return pt;
  }

  // 直接评估当前段中某个 u ∈ [0,1] 上的值
  Eigen::Vector2d evaluate(double u_eval) const {
    if (!segment_ready)
      return Eigen::Vector2d::Zero();
    return deBoorEvaluate(u_eval);
  }

private:
  int k;                    // B样条阶数
  double u = 0.0;           // 当前段内采样位置
  double delta_u;           // 采样间隔
  size_t current_index = 0; // 当前段起始控制点索引
  bool segment_ready = false;

  std::vector<Eigen::Vector2d> full_points;    // 全部控制点
  std::deque<Eigen::Vector2d> current_segment; // 当前段的 k+1 控制点
  std::vector<double> knots;                   // 全局节点向量

  // 构建当前段
  void updateSegment() {
    segment_ready = false;
    if (current_index + k >= full_points.size())
      return;

    current_segment.clear();
    for (int i = 0; i <= k; ++i)
      current_segment.push_back(full_points[current_index + i]);

    segment_ready = true;
  }

  // 生成 clamped 全局均匀 knot 向量
  void generateGlobalKnots() {
    int n = static_cast<int>(full_points.size()) - 1;
    int m = n + k + 1;
    knots.resize(m + 1);

    for (int i = 0; i <= m; ++i) {
      if (i <= k)
        knots[i] = 0.0;
      else if (i >= m - k)
        knots[i] = 1.0;
      else
        knots[i] = double(i - k) / (m - 2 * k);
    }
  }

  // 非递归 de Boor 基函数计算（提升效率）
  std::vector<double> computeBasis(double u_eval) const {
    std::vector<double> N(k + 1, 0.0);
    std::vector<double> left(k + 1), right(k + 1);
    N[0] = 1.0;

    for (int j = 1; j <= k; ++j) {
      left[j] = u_eval - knots[current_index + k + 1 - j];
      right[j] = knots[current_index + k + j] - u_eval;

      double saved = 0.0;
      for (int r = 0; r < j; ++r) {
        double denom = right[r + 1] + left[j - r];
        double temp = (denom > 1e-6) ? N[r] / denom : 0.0;
        N[r] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      N[j] = saved;
    }

    return N;
  }

  // de Boor 插值点计算
  Eigen::Vector2d deBoorEvaluate(double u_eval) const {
    Eigen::Vector2d result = Eigen::Vector2d::Zero();
    auto basis = computeBasis(u_eval);
    for (int i = 0; i <= k; ++i)
      result += basis[i] * current_segment[i];
    return result;
  }

  // 简单滑动窗口降噪（Savitzky-Golay 可替代）
  std::vector<Eigen::Vector2d>
  smoothPoints(const std::vector<Eigen::Vector2d> &pts, int win = 3) const {
    if (pts.size() < 2 * win + 1)
      return pts;

    std::vector<Eigen::Vector2d> smoothed = pts;
    for (size_t i = win; i + win < pts.size(); ++i) {
      Eigen::Vector2d avg = Eigen::Vector2d::Zero();
      for (int j = -win; j <= win; ++j)
        avg += pts[i + j];
      smoothed[i] = avg / (2 * win + 1);
    }
    return smoothed;
  }
};
