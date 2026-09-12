#include "ContourBuilder.h"

#include <QLineF>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ContourBuilder
{
namespace
{

struct Point
{
    QPointF position;
    QPointF tangent;
    bool hasTangent = false;
};

struct Edge
{
    Edge(int first, int second, double value) : a(first), b(second), cost(value) {}
    int a;
    int b;
    double cost;
};

struct CellKey
{
    int x = 0;
    int y = 0;

    bool operator==(const CellKey& other) const { return x == other.x && y == other.y; }
};

struct CellHash
{
    size_t operator()(const CellKey& key) const
    {
        const size_t x = static_cast<size_t>(static_cast<uint32_t>(key.x));
        const size_t y = static_cast<size_t>(static_cast<uint32_t>(key.y));
        return x * static_cast<size_t>(0x9e3779b9U) ^ (y + (x << 6U) + (x >> 2U));
    }
};

double dot(const QPointF& lhs, const QPointF& rhs)
{
    return lhs.x() * rhs.x() + lhs.y() * rhs.y();
}

double norm(const QPointF& value)
{
    return std::sqrt(dot(value, value));
}

QPointF normalized(const QPointF& value)
{
    const double length = norm(value);
    if (length <= 1e-9) return {};
    return value / length;
}

CellKey cellOf(const QPointF& point, double cellSize)
{
    CellKey key;
    key.x = static_cast<int>(std::floor(point.x() / cellSize));
    key.y = static_cast<int>(std::floor(point.y() / cellSize));
    return key;
}

double distanceSquared(const QPointF& lhs, const QPointF& rhs)
{
    const double dx = lhs.x() - rhs.x();
    const double dy = lhs.y() - rhs.y();
    return dx * dx + dy * dy;
}

bool hasProperIntersection(const QPointF& a, const QPointF& b,
                           const QPointF& c, const QPointF& d)
{
    const QLineF first(a, b);
    const QLineF second(c, d);
    if (first.p1() == second.p1() || first.p1() == second.p2() ||
        first.p2() == second.p1() || first.p2() == second.p2()) return false;
    QPointF intersection;
    return first.intersects(second, &intersection) == QLineF::BoundedIntersection;
}

double nearestSpacing(const std::vector<Point>& points,
                      const std::unordered_map<CellKey, std::vector<int>, CellHash>& cells,
                      double cellSize)
{
    std::vector<double> nearest;
    nearest.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double best = std::numeric_limits<double>::infinity();
        const CellKey center = cellOf(points[i].position, cellSize);
        for (int radius = 1; radius <= 6 && !std::isfinite(best); ++radius) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    CellKey key;
                    key.x = center.x + dx;
                    key.y = center.y + dy;
                    const auto it = cells.find(key);
                    if (it == cells.end()) continue;
                    for (const int j : it->second) {
                        if (i == j) continue;
                        best = std::min(best, std::sqrt(distanceSquared(
                            points[i].position, points[j].position)));
                    }
                }
            }
        }
        if (std::isfinite(best) && best > 1e-6) nearest.push_back(best);
    }
    if (nearest.empty()) return 1.0;
    std::nth_element(nearest.begin(), nearest.begin() + nearest.size() / 2, nearest.end());
    return std::max(0.25, nearest[nearest.size() / 2]);
}

QVector<ImageView::ContourPath> traceFeatureGraph(const std::vector<Point>& points,
                                                   double spacing)
{
    if (points.size() < 2) return {};

    const double cellSize = std::max(1.0, spacing * 1.5);
    const double maxJoinDistance = std::min(10.0, std::max(2.5, spacing * 3.5));
    std::unordered_map<CellKey, std::vector<int>, CellHash> cells;
    cells.reserve(points.size() * 2 + 1);
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        cells[cellOf(points[i].position, cellSize)].push_back(i);

    std::vector<Edge> candidates;
    candidates.reserve(points.size() * 6);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const CellKey center = cellOf(points[i].position, cellSize);
        std::vector<Edge> local;
        const int cellRadius = static_cast<int>(std::ceil(maxJoinDistance / cellSize));
        for (int dy = -cellRadius; dy <= cellRadius; ++dy) {
            for (int dx = -cellRadius; dx <= cellRadius; ++dx) {
                CellKey key;
                key.x = center.x + dx;
                key.y = center.y + dy;
                const auto it = cells.find(key);
                if (it == cells.end()) continue;
                for (const int j : it->second) {
                    if (j <= i) continue;
                    const double distance = std::sqrt(distanceSquared(
                        points[i].position, points[j].position));
                    if (distance <= 1e-6 || distance > maxJoinDistance) continue;
                    const QPointF direction = (points[j].position - points[i].position) / distance;
                    double alignment = 0.0;
                    int tangentCount = 0;
                    if (points[i].hasTangent) {
                        alignment += std::abs(dot(points[i].tangent, direction));
                        ++tangentCount;
                    }
                    if (points[j].hasTangent) {
                        alignment += std::abs(dot(points[j].tangent, direction));
                        ++tangentCount;
                    }
                    if (tangentCount > 0) alignment /= tangentCount;
                    const double cost = distance / spacing +
                        (tangentCount > 0 ? 1.5 * (1.0 - alignment) : 0.0);
                    local.push_back(Edge(i, j, cost));
                }
            }
        }
        std::sort(local.begin(), local.end(), [](const Edge& lhs, const Edge& rhs) {
            return lhs.cost < rhs.cost;
        });
        const size_t limit = std::min<size_t>(local.size(), 8);
        candidates.insert(candidates.end(), local.begin(), local.begin() + limit);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.cost < rhs.cost;
    });

    // 先隔离空间连通的轮廓分量，再对每个分量独立选边，避免稠密外边界的度数饱和
    // 消耗孔洞邻接点。
    std::vector<int> parent(points.size());
    std::vector<int> rank(points.size(), 0);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) parent[i] = i;
    const auto findRoot = [&](int value) {
        int root = value;
        while (parent[root] != root) root = parent[root];
        return root;
    };
    const auto unite = [&](int lhs, int rhs) {
        int left = findRoot(lhs), right = findRoot(rhs);
        if (left == right) return;
        if (rank[left] < rank[right]) std::swap(left, right);
        parent[right] = left;
        if (rank[left] == rank[right]) ++rank[left];
    };
    for (const Edge& candidate : candidates) unite(candidate.a, candidate.b);
    std::vector<std::vector<int>> componentCandidates(points.size());
    for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
        componentCandidates[findRoot(candidates[index].a)].push_back(index);

    std::vector<std::vector<int>> rankedNeighbors(points.size());
    for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
        rankedNeighbors[candidates[index].a].push_back(index);
        rankedNeighbors[candidates[index].b].push_back(index);
    }
    for (auto& neighbors : rankedNeighbors)
        std::sort(neighbors.begin(), neighbors.end(), [&](int lhs, int rhs) {
            return candidates[lhs].cost < candidates[rhs].cost;
        });

    std::vector<std::vector<int>> adjacency(points.size());
    std::vector<Edge> accepted;
    const auto isMutualNearest = [&](int candidateIndex) {
        const Edge& candidate = candidates[candidateIndex];
        const auto hasNear = [&](int node, int wanted) {
            const std::vector<int>& neighbors = rankedNeighbors[node];
            const size_t limit = std::min<size_t>(3, neighbors.size());
            for (size_t i = 0; i < limit; ++i) {
                const Edge& edge = candidates[neighbors[i]];
                const int other = edge.a == node ? edge.b : edge.a;
                if (other == wanted) return true;
            }
            return false;
        };
        return hasNear(candidate.a, candidate.b) && hasNear(candidate.b, candidate.a);
    };
    const auto acceptCandidate = [&](const Edge& candidate) {
        if (adjacency[candidate.a].size() >= 2 || adjacency[candidate.b].size() >= 2)
            return;
        for (const Edge& existing : accepted) {
            if (hasProperIntersection(points[candidate.a].position,
                                       points[candidate.b].position,
                                       points[existing.a].position,
                                       points[existing.b].position)) return;
        }
        accepted.push_back(candidate);
        adjacency[candidate.a].push_back(candidate.b);
        adjacency[candidate.b].push_back(candidate.a);
    };
    for (const auto& component : componentCandidates) {
        if (component.empty()) continue;
        // 先用互为最近邻的连接建立稳定骨架。
        for (const int index : component)
            if (isMutualNearest(index)) acceptCandidate(candidates[index]);
        // 剩余连接用于跨越合法稀疏间隙，但不允许分量借用其他轮廓的边。
        for (const int index : component)
            if (!isMutualNearest(index)) acceptCandidate(candidates[index]);
    }

    QVector<ImageView::ContourPath> result;
    std::vector<bool> visited(points.size(), false);
    const auto trace = [&](int start, bool cycle) {
        QVector<QPointF> path;
        int previous = -1;
        int current = start;
        while (current >= 0 && !visited[current]) {
            visited[current] = true;
            path.push_back(points[current].position);
            int next = -1;
            for (const int neighbor : adjacency[current]) {
                if (neighbor == previous) continue;
                if (cycle && neighbor == start && path.size() > 2) {
                    next = -1;
                    break;
                }
                if (!visited[neighbor]) {
                    next = neighbor;
                    break;
                }
            }
            if (next < 0) break;
            previous = current;
            current = next;
        }
        if (path.size() < 2) return;
        ImageView::ContourPath contour;
        contour.points = QPolygonF(path);
        contour.closed = cycle && path.size() >= 3;
        result.push_back(contour);
    };

    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        if (!visited[i] && adjacency[i].size() == 1) trace(i, false);
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        if (!visited[i] && adjacency[i].size() >= 2) trace(i, true);
    return result;
}

}

QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model)
{
    return buildTemplateContours(image, model, 0);
}

QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model, int level)
{
    if (image.empty() || !model || level < 0 ||
        level >= static_cast<int>(model->templates.size()) || !model->templates[level] ||
        model->templates[level]->shape_angle.empty() ||
        !model->templates[level]->shape_angle[0]) return {};
    if (image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3 &&
                                  image.channels() != 4)) return {};

    const std::vector<T_T::ShapePoint>& features =
        model->templates[level]->shape_angle[0]->shape_point;
    if (features.empty()) return {};

    std::vector<Point> points;
    points.reserve(features.size());
    double originX = image.cols * 0.5;
    double originY = image.rows * 0.5;
    if (model->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID) {
        const double scale = static_cast<double>(1 << level);
        originX = model->template_cfg.origin_x / scale;
        originY = model->template_cfg.origin_y / scale;
    }
    for (const T_T::ShapePoint& feature : features) {
        const QPointF position(originX + feature.x, originY + feature.y);
        if (position.x() < 0.0 || position.y() < 0.0 ||
            position.x() >= image.cols || position.y() >= image.rows) continue;
        const QPointF gradient(feature.edge_dx, feature.edge_dy);
        const double gradientLength = norm(gradient);
        Point point;
        point.position = position;
        if (gradientLength > 1e-6)
            point.tangent = normalized(QPointF(-gradient.y(), gradient.x()));
        point.hasTangent = gradientLength > 1e-6;
        points.push_back(point);
    }
    if (points.size() < 2) return {};

    // ShapePoint 已经是单个定位后的边缘采样点；这里不使用栅格掩模、形态学或
    // findContours，因此单像素边缘环不会被变成厚栅格轮廓的两条边。
    const double cellSize = 1.5;
    std::unordered_map<CellKey, std::vector<int>, CellHash> cells;
    cells.reserve(points.size() * 2 + 1);
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        cells[cellOf(points[i].position, cellSize)].push_back(i);
    const double spacing = nearestSpacing(points, cells, cellSize);
    return traceFeatureGraph(points, spacing);
}

}
