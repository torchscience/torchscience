#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace torchscience::kernel::geometry {

// ============================================================================
// 2D Convex Hull using Graham Scan
// ============================================================================

template <typename scalar_t>
struct ConvexHull2D {
  std::vector<int64_t> vertices;    // Vertex indices in CCW order
  std::vector<scalar_t> equations;  // Facet equations: (nx, ny, d) per edge
  scalar_t perimeter;               // Perimeter (area in N-1 dimension)
  scalar_t area;                    // Signed area

  // Cross product of vectors (p1-p0) and (p2-p0)
  static inline scalar_t cross(const scalar_t* pts, int64_t p0, int64_t p1,
                               int64_t p2) {
    scalar_t x0 = pts[p0 * 2], y0 = pts[p0 * 2 + 1];
    scalar_t x1 = pts[p1 * 2], y1 = pts[p1 * 2 + 1];
    scalar_t x2 = pts[p2 * 2], y2 = pts[p2 * 2 + 1];
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  }

  void compute(const scalar_t* points, int64_t n) {
    if (n < 3) {
      vertices.clear();
      for (int64_t i = 0; i < n; ++i) vertices.push_back(i);
      perimeter = scalar_t(0);
      area = scalar_t(0);
      return;
    }

    // Find bottom-left point (pivot)
    int64_t pivot = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (points[i * 2 + 1] < points[pivot * 2 + 1] ||
          (points[i * 2 + 1] == points[pivot * 2 + 1] &&
           points[i * 2] < points[pivot * 2])) {
        pivot = i;
      }
    }

    // Sort points by polar angle relative to pivot
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) indices[i] = i;
    std::swap(indices[0], indices[pivot]);

    const scalar_t px = points[indices[0] * 2];
    const scalar_t py = points[indices[0] * 2 + 1];

    std::sort(indices.begin() + 1, indices.end(), [&](int64_t a, int64_t b) {
      scalar_t ax = points[a * 2] - px;
      scalar_t ay = points[a * 2 + 1] - py;
      scalar_t bx = points[b * 2] - px;
      scalar_t by = points[b * 2 + 1] - py;
      scalar_t cross_val = ax * by - ay * bx;
      if (std::abs(cross_val) < scalar_t(1e-10)) {
        // Collinear: sort by distance
        return (ax * ax + ay * ay) < (bx * bx + by * by);
      }
      return cross_val > 0;
    });

    // Graham scan
    vertices.clear();
    for (int64_t i = 0; i < n; ++i) {
      while (vertices.size() > 1 &&
             cross(points, vertices[vertices.size() - 2],
                   vertices[vertices.size() - 1], indices[i]) <= 0) {
        vertices.pop_back();
      }
      vertices.push_back(indices[i]);
    }

    // Compute area and perimeter
    int64_t m = static_cast<int64_t>(vertices.size());
    area = scalar_t(0);
    perimeter = scalar_t(0);
    equations.resize(m * 3);

    for (int64_t i = 0; i < m; ++i) {
      int64_t j = (i + 1) % m;
      int64_t vi = vertices[i];
      int64_t vj = vertices[j];

      scalar_t x0 = points[vi * 2], y0 = points[vi * 2 + 1];
      scalar_t x1 = points[vj * 2], y1 = points[vj * 2 + 1];

      // Shoelace formula
      area += x0 * y1 - x1 * y0;

      // Edge length
      scalar_t dx = x1 - x0, dy = y1 - y0;
      scalar_t len = std::sqrt(dx * dx + dy * dy);
      perimeter += len;

      // Edge equation: outward normal and offset
      if (len > scalar_t(1e-10)) {
        scalar_t nx = dy / len;  // Outward normal (perpendicular to edge)
        scalar_t ny = -dx / len;
        scalar_t d = -(nx * x0 + ny * y0);
        equations[i * 3 + 0] = nx;
        equations[i * 3 + 1] = ny;
        equations[i * 3 + 2] = d;
      } else {
        equations[i * 3 + 0] = scalar_t(0);
        equations[i * 3 + 1] = scalar_t(0);
        equations[i * 3 + 2] = scalar_t(0);
      }
    }
    area = std::abs(area) / scalar_t(2);
  }
};

// ============================================================================
// 3D Convex Hull using Quickhull
// ============================================================================

template <typename scalar_t>
struct ConvexHull3D {
  std::vector<int64_t> vertices;
  std::vector<int64_t> simplices;   // Triangular facets (3 vertices each)
  std::vector<int64_t> neighbors;   // Neighbor facet indices
  std::vector<scalar_t> equations;  // (nx, ny, nz, d) per facet
  scalar_t surface_area;
  scalar_t volume;

  // Facet structure for Quickhull
  struct Facet {
    int64_t v[3];           // Vertex indices
    int64_t neighbors[3];   // Neighbor facet indices
    scalar_t normal[3];     // Unit normal
    scalar_t offset;        // Plane offset
    std::vector<int64_t> outside_set;
    int64_t furthest_point;
    scalar_t furthest_dist;
    bool active;

    void compute_plane(const scalar_t* pts) {
      // Compute normal from cross product
      scalar_t ax = pts[v[1] * 3 + 0] - pts[v[0] * 3 + 0];
      scalar_t ay = pts[v[1] * 3 + 1] - pts[v[0] * 3 + 1];
      scalar_t az = pts[v[1] * 3 + 2] - pts[v[0] * 3 + 2];
      scalar_t bx = pts[v[2] * 3 + 0] - pts[v[0] * 3 + 0];
      scalar_t by = pts[v[2] * 3 + 1] - pts[v[0] * 3 + 1];
      scalar_t bz = pts[v[2] * 3 + 2] - pts[v[0] * 3 + 2];

      normal[0] = ay * bz - az * by;
      normal[1] = az * bx - ax * bz;
      normal[2] = ax * by - ay * bx;

      scalar_t len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                               normal[2] * normal[2]);

      if (len > scalar_t(1e-10)) {
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
      }

      offset = -(normal[0] * pts[v[0] * 3 + 0] + normal[1] * pts[v[0] * 3 + 1] +
                 normal[2] * pts[v[0] * 3 + 2]);
    }

    scalar_t signed_distance(const scalar_t* pt) const {
      return normal[0] * pt[0] + normal[1] * pt[1] + normal[2] * pt[2] + offset;
    }
  };

  void compute(const scalar_t* points, int64_t n) {
    if (n < 4) {
      // Degenerate: return all points as vertices
      vertices.clear();
      for (int64_t i = 0; i < n; ++i) vertices.push_back(i);
      surface_area = scalar_t(0);
      volume = scalar_t(0);
      return;
    }

    // Step 1: Find initial tetrahedron
    std::vector<int64_t> initial = find_initial_simplex(points, n);
    if (initial.size() < 4) {
      // Degenerate (coplanar)
      vertices.assign(initial.begin(), initial.end());
      surface_area = scalar_t(0);
      volume = scalar_t(0);
      return;
    }

    // Step 2: Initialize with 4 facets of tetrahedron
    std::vector<Facet> facets;
    facets.reserve(n * 4);  // Upper bound estimate

    // Create 4 faces of tetrahedron (ensure outward normals)
    int64_t faces[4][3] = {{initial[0], initial[1], initial[2]},
                           {initial[0], initial[2], initial[3]},
                           {initial[0], initial[3], initial[1]},
                           {initial[1], initial[3], initial[2]}};

    // Compute centroid for orientation
    scalar_t cx = scalar_t(0), cy = scalar_t(0), cz = scalar_t(0);
    for (int i = 0; i < 4; ++i) {
      cx += points[initial[i] * 3 + 0];
      cy += points[initial[i] * 3 + 1];
      cz += points[initial[i] * 3 + 2];
    }
    cx /= 4;
    cy /= 4;
    cz /= 4;

    for (int i = 0; i < 4; ++i) {
      Facet f;
      f.v[0] = faces[i][0];
      f.v[1] = faces[i][1];
      f.v[2] = faces[i][2];
      f.active = true;
      f.furthest_dist = scalar_t(-1);
      f.furthest_point = -1;
      f.compute_plane(points);

      // Ensure normal points outward (away from centroid)
      scalar_t center[3] = {cx, cy, cz};
      if (f.signed_distance(center) > 0) {
        // Flip normal and swap vertices
        std::swap(f.v[1], f.v[2]);
        f.compute_plane(points);
      }

      facets.push_back(f);
    }

    // Set up neighbor relationships for initial tetrahedron
    facets[0].neighbors[0] = 3;
    facets[0].neighbors[1] = 2;
    facets[0].neighbors[2] = 1;
    facets[1].neighbors[0] = 3;
    facets[1].neighbors[1] = 0;
    facets[1].neighbors[2] = 2;
    facets[2].neighbors[0] = 3;
    facets[2].neighbors[1] = 1;
    facets[2].neighbors[2] = 0;
    facets[3].neighbors[0] = 1;
    facets[3].neighbors[1] = 2;
    facets[3].neighbors[2] = 0;

    // Step 3: Assign all points to outside sets
    std::vector<bool> in_hull(n, false);
    for (int i = 0; i < 4; ++i) in_hull[initial[i]] = true;

    for (int64_t i = 0; i < n; ++i) {
      if (in_hull[i]) continue;

      scalar_t max_dist = scalar_t(-1e10);
      int64_t best_facet = -1;

      for (size_t f = 0; f < facets.size(); ++f) {
        if (!facets[f].active) continue;
        scalar_t dist = facets[f].signed_distance(&points[i * 3]);
        if (dist > scalar_t(1e-10) && dist > max_dist) {
          max_dist = dist;
          best_facet = static_cast<int64_t>(f);
        }
      }

      if (best_facet >= 0) {
        facets[best_facet].outside_set.push_back(i);
        if (max_dist > facets[best_facet].furthest_dist) {
          facets[best_facet].furthest_dist = max_dist;
          facets[best_facet].furthest_point = i;
        }
      }
    }

    // Step 4: Iteratively add points
    bool changed = true;
    while (changed) {
      changed = false;

      for (size_t f = 0; f < facets.size(); ++f) {
        if (!facets[f].active || facets[f].outside_set.empty()) continue;

        // Get apex (furthest point)
        int64_t apex = facets[f].furthest_point;

        // Find all visible facets from apex
        std::vector<int64_t> visible;
        std::vector<bool> visited(facets.size(), false);
        find_visible_facets(facets, f, points, apex, visible, visited);

        if (visible.empty()) continue;

        // Find horizon (boundary between visible and invisible)
        std::vector<std::pair<int64_t, int64_t>> horizon;
        find_horizon(facets, visible, horizon);

        // Create new facets connecting apex to horizon
        int64_t first_new = static_cast<int64_t>(facets.size());
        for (const auto& edge : horizon) {
          Facet nf;
          nf.v[0] = apex;
          nf.v[1] = edge.first;
          nf.v[2] = edge.second;
          nf.active = true;
          nf.furthest_dist = scalar_t(-1);
          nf.furthest_point = -1;
          nf.compute_plane(points);

          // Ensure outward normal
          scalar_t dist_check = nf.signed_distance(&points[initial[0] * 3]);
          if (dist_check > scalar_t(1e-10)) {
            std::swap(nf.v[1], nf.v[2]);
            nf.compute_plane(points);
          }

          nf.neighbors[0] = -1;
          nf.neighbors[1] = -1;
          nf.neighbors[2] = -1;
          facets.push_back(nf);
        }

        // Deactivate visible facets and redistribute outside sets
        std::vector<int64_t> points_to_redistribute;
        for (int64_t vi : visible) {
          facets[vi].active = false;
          for (int64_t pi : facets[vi].outside_set) {
            if (pi != apex) {
              points_to_redistribute.push_back(pi);
            }
          }
          facets[vi].outside_set.clear();
        }

        // Redistribute points to new facets
        for (int64_t pi : points_to_redistribute) {
          scalar_t max_dist = scalar_t(-1e10);
          int64_t best_facet = -1;

          for (size_t nf = first_new; nf < facets.size(); ++nf) {
            if (!facets[nf].active) continue;
            scalar_t dist = facets[nf].signed_distance(&points[pi * 3]);
            if (dist > scalar_t(1e-10) && dist > max_dist) {
              max_dist = dist;
              best_facet = static_cast<int64_t>(nf);
            }
          }

          if (best_facet >= 0) {
            facets[best_facet].outside_set.push_back(pi);
            if (max_dist > facets[best_facet].furthest_dist) {
              facets[best_facet].furthest_dist = max_dist;
              facets[best_facet].furthest_point = pi;
            }
          }
        }

        changed = true;
        break;  // Restart loop since facet indices changed
      }
    }

    // Step 5: Extract results
    std::set<int64_t> vertex_set;
    simplices.clear();
    equations.clear();
    neighbors.clear();

    for (const auto& f : facets) {
      if (!f.active) continue;
      for (int i = 0; i < 3; ++i) {
        vertex_set.insert(f.v[i]);
      }
      simplices.push_back(f.v[0]);
      simplices.push_back(f.v[1]);
      simplices.push_back(f.v[2]);
      equations.push_back(f.normal[0]);
      equations.push_back(f.normal[1]);
      equations.push_back(f.normal[2]);
      equations.push_back(f.offset);
      neighbors.push_back(f.neighbors[0]);
      neighbors.push_back(f.neighbors[1]);
      neighbors.push_back(f.neighbors[2]);
    }

    vertices.assign(vertex_set.begin(), vertex_set.end());

    // Compute surface area and volume
    compute_metrics(points);
  }

 private:
  std::vector<int64_t> find_initial_simplex(const scalar_t* pts, int64_t n) {
    std::vector<int64_t> result;

    // Find extremal points along each axis
    int64_t min_x = 0, max_x = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (pts[i * 3] < pts[min_x * 3]) min_x = i;
      if (pts[i * 3] > pts[max_x * 3]) max_x = i;
    }

    if (min_x == max_x) return result;  // All points coincident

    result.push_back(min_x);
    result.push_back(max_x);

    // Find point furthest from line
    scalar_t max_dist = scalar_t(0);
    int64_t third = -1;
    for (int64_t i = 0; i < n; ++i) {
      if (i == min_x || i == max_x) continue;
      scalar_t dist =
          point_line_dist(&pts[i * 3], &pts[min_x * 3], &pts[max_x * 3]);
      if (dist > max_dist) {
        max_dist = dist;
        third = i;
      }
    }

    if (third < 0) return result;  // Collinear
    result.push_back(third);

    // Find point furthest from plane
    max_dist = scalar_t(0);
    int64_t fourth = -1;

    // Compute plane normal
    scalar_t ax = pts[max_x * 3] - pts[min_x * 3];
    scalar_t ay = pts[max_x * 3 + 1] - pts[min_x * 3 + 1];
    scalar_t az = pts[max_x * 3 + 2] - pts[min_x * 3 + 2];
    scalar_t bx = pts[third * 3] - pts[min_x * 3];
    scalar_t by = pts[third * 3 + 1] - pts[min_x * 3 + 1];
    scalar_t bz = pts[third * 3 + 2] - pts[min_x * 3 + 2];
    scalar_t nx = ay * bz - az * by;
    scalar_t ny = az * bx - ax * bz;
    scalar_t nz = ax * by - ay * bx;
    scalar_t len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > scalar_t(1e-10)) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    scalar_t d = -(nx * pts[min_x * 3] + ny * pts[min_x * 3 + 1] +
                   nz * pts[min_x * 3 + 2]);

    for (int64_t i = 0; i < n; ++i) {
      if (i == min_x || i == max_x || i == third) continue;
      scalar_t dist = std::abs(nx * pts[i * 3] + ny * pts[i * 3 + 1] +
                               nz * pts[i * 3 + 2] + d);
      if (dist > max_dist) {
        max_dist = dist;
        fourth = i;
      }
    }

    if (fourth < 0) return result;  // Coplanar
    result.push_back(fourth);

    return result;
  }

  scalar_t point_line_dist(const scalar_t* p, const scalar_t* a,
                           const scalar_t* b) {
    scalar_t abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    scalar_t apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
    // Cross product ap x ab
    scalar_t cx = apy * abz - apz * aby;
    scalar_t cy = apz * abx - apx * abz;
    scalar_t cz = apx * aby - apy * abx;
    scalar_t cross_len = std::sqrt(cx * cx + cy * cy + cz * cz);
    scalar_t ab_len = std::sqrt(abx * abx + aby * aby + abz * abz);
    if (ab_len < scalar_t(1e-10)) return scalar_t(0);
    return cross_len / ab_len;
  }

  void find_visible_facets(std::vector<Facet>& facets, int64_t start,
                           const scalar_t* points, int64_t apex,
                           std::vector<int64_t>& visible,
                           std::vector<bool>& visited) {
    if (visited[start] || !facets[start].active) return;
    visited[start] = true;

    if (facets[start].signed_distance(&points[apex * 3]) > scalar_t(1e-10)) {
      visible.push_back(start);
      for (int i = 0; i < 3; ++i) {
        int64_t nb = facets[start].neighbors[i];
        if (nb >= 0 && !visited[nb]) {
          find_visible_facets(facets, nb, points, apex, visible, visited);
        }
      }
    }
  }

  void find_horizon(const std::vector<Facet>& facets,
                    const std::vector<int64_t>& visible,
                    std::vector<std::pair<int64_t, int64_t>>& horizon) {
    std::set<int64_t> visible_set(visible.begin(), visible.end());

    for (int64_t fi : visible) {
      const Facet& f = facets[fi];
      for (int i = 0; i < 3; ++i) {
        int64_t nb = f.neighbors[i];
        if (nb < 0 || visible_set.find(nb) == visible_set.end()) {
          // This edge is on the horizon
          int64_t e1 = f.v[(i + 1) % 3];
          int64_t e2 = f.v[(i + 2) % 3];
          horizon.emplace_back(e1, e2);
        }
      }
    }
  }

  void compute_metrics(const scalar_t* points) {
    surface_area = scalar_t(0);
    volume = scalar_t(0);

    int64_t n_facets = static_cast<int64_t>(simplices.size()) / 3;
    if (n_facets == 0) return;

    // Reference point for volume calculation (any vertex works)
    int64_t ref = simplices[0];
    scalar_t rx = points[ref * 3], ry = points[ref * 3 + 1],
             rz = points[ref * 3 + 2];

    for (int64_t i = 0; i < n_facets; ++i) {
      int64_t v0 = simplices[i * 3];
      int64_t v1 = simplices[i * 3 + 1];
      int64_t v2 = simplices[i * 3 + 2];

      // Triangle area via cross product
      scalar_t ax = points[v1 * 3] - points[v0 * 3];
      scalar_t ay = points[v1 * 3 + 1] - points[v0 * 3 + 1];
      scalar_t az = points[v1 * 3 + 2] - points[v0 * 3 + 2];
      scalar_t bx = points[v2 * 3] - points[v0 * 3];
      scalar_t by = points[v2 * 3 + 1] - points[v0 * 3 + 1];
      scalar_t bz = points[v2 * 3 + 2] - points[v0 * 3 + 2];

      scalar_t cx = ay * bz - az * by;
      scalar_t cy = az * bx - ax * bz;
      scalar_t cz = ax * by - ay * bx;

      scalar_t tri_area = std::sqrt(cx * cx + cy * cy + cz * cz) / scalar_t(2);
      surface_area += tri_area;

      // Tetrahedron volume (reference point to triangle)
      scalar_t dx = points[v0 * 3] - rx;
      scalar_t dy = points[v0 * 3 + 1] - ry;
      scalar_t dz = points[v0 * 3 + 2] - rz;

      // Volume = (1/6) * |d . (a x b)|
      scalar_t tet_vol = (dx * cx + dy * cy + dz * cz) / scalar_t(6);
      volume += tet_vol;
    }

    volume = std::abs(volume);
  }
};

}  // namespace torchscience::kernel::geometry
