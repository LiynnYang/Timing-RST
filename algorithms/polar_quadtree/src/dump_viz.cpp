#include "polar_quadtree/framework.h"
#include "polar_quadtree/geometry.h"
#include "polar_quadtree/merge.h"
#include "polar_quadtree/quadtree.h"
#include "polar_quadtree/solvers.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

void write_point(std::ostream& os, const pqt::Point& p) {
    os << "{\"x\":" << p.x << ",\"y\":" << p.y << "}";
}

std::string json_escape(const std::string&) { return ""; }

}  // namespace

int main(int argc, char** argv) {
    const int n = 48;
    const int capacity = 8;
    std::vector<pqt::Point> points(static_cast<std::size_t>(n));
    points[0] = {0.0, 0.0};
    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (int i = 1; i < n; ++i) {
        points[static_cast<std::size_t>(i)] = {dist(rng), dist(rng)};
    }

    const pqt::Point origin = points[0];
    std::vector<pqt::Point> translated;
    translated.reserve(points.size());
    for (const auto& p : points) {
        translated.push_back(pqt::translate(p, origin));
    }

    pqt::PolarQuadtree qt = pqt::build_polar_quadtree(translated, capacity);
    const double bound = qt.default_search_bound;
    pqt::compute_merge_topology(qt, bound);
    std::vector<pqt::InterBlockLink> links;
    pqt::assign_sources_and_links(qt, translated, origin, links);

    const pqt::Tree final_tree =
        pqt::divide_and_merge(points, pqt::rectilinear_mst, capacity);

    const char* out_path = argc > 1 ? argv[1] : "viz/pipeline.json";
    std::ofstream os(out_path);
    if (!os) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }

    os << "{\n  \"origin\": ";
    write_point(os, origin);
    os << ",\n  \"points\": [";
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (i) {
            os << ", ";
        }
        write_point(os, points[i]);
    }

    os << "],\n  \"blocks\": [";
    bool first_block = true;
    for (const auto& b : qt.blocks) {
        if (b.layer < 0) {
            continue;
        }
        if (!first_block) {
            os << ", ";
        }
        first_block = false;
        os << "{\"id\":" << b.id << ",\"layer\":" << b.layer << ",\"index\":" << b.index
           << ",\"r_inner\":" << b.r_inner << ",\"r_outer\":" << b.r_outer
           << ",\"theta_min\":" << b.theta_min << ",\"theta_max\":" << b.theta_max
           << ",\"nonempty\":" << (b.nonempty() ? "true" : "false")
           << ",\"local_source\":" << b.local_source << ",\"pins\":[";
        for (std::size_t i = 0; i < b.pin_ids.size(); ++i) {
            if (i) {
                os << ",";
            }
            os << b.pin_ids[i];
        }
        os << "]}";
    }

    os << "],\n  \"subtrees\": [";
    bool first_tree = true;
    for (const auto& b : qt.blocks) {
        if (!b.nonempty() || b.local_source < 0) {
            continue;
        }
        std::vector<pqt::Point> local;
        local.push_back(points[static_cast<std::size_t>(b.local_source)]);
        for (int pid : b.pin_ids) {
            if (pid != b.local_source) {
                local.push_back(points[static_cast<std::size_t>(pid)]);
            }
        }
        pqt::Tree sub = local.size() <= 1 ? pqt::Tree{local, {}}
                                          : pqt::rectilinear_mst(local);
        if (!first_tree) {
            os << ", ";
        }
        first_tree = false;
        os << "{\"block\":" << b.id << ",\"nodes\":[";
        for (std::size_t i = 0; i < sub.nodes.size(); ++i) {
            if (i) {
                os << ",";
            }
            write_point(os, sub.nodes[i]);
        }
        os << "],\"edges\":[";
        for (std::size_t i = 0; i < sub.edges.size(); ++i) {
            if (i) {
                os << ",";
            }
            os << "[" << sub.edges[i].first << "," << sub.edges[i].second << "]";
        }
        os << "]}";
    }

    os << "],\n  \"final\": {\"nodes\":[";
    for (std::size_t i = 0; i < final_tree.nodes.size(); ++i) {
        if (i) {
            os << ",";
        }
        write_point(os, final_tree.nodes[i]);
    }
    os << "],\"edges\":[";
    for (std::size_t i = 0; i < final_tree.edges.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << "[" << final_tree.edges[i].first << "," << final_tree.edges[i].second
           << "]";
    }
    os << "]}\n}\n";
    (void)json_escape;
    return 0;
}
