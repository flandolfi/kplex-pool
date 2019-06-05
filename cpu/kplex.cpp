#include <torch/extension.h>
#include "comparer.hpp"


enum class NodePriority { 
    RANDOM, 
    MIN_DEGREE,
    MAX_DEGREE, 
    MIN_UNCOVERED, 
    MAX_UNCOVERED, 
    MIN_IN_KPLEX,
    MAX_IN_KPLEX,
    MIN_CANDIDATES,
    MAX_CANDIDATES,
    COMPOSITE
};

/* From torch_cluster/cpu/utils.h by @rusty1s */
std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row, at::Tensor col) {
    auto mask = row != col;
    return std::make_tuple(row.masked_select(mask), col.masked_select(mask));
}

std::unordered_set<int64_t> find_kplex(const std::vector<std::unordered_set<int64_t>>& neighbors, 
            int64_t node, int64_t k, int64_t num_nodes, std::vector<bool>& covered_nodes, 
            NodePriority kplex_priority, std::vector<int64_t>& priorities, bool skip_covered,
            const std::function<void(int64_t)>& node_callback) {
    std::unordered_set<int64_t> excluded({node});
    std::unordered_set<int64_t> kplex({node});
    std::unordered_map<int64_t, int64_t> missing_links({{node, 1}});
    std::unordered_map<int64_t, int64_t> candidate_links;
    PriorityComparer<std::unordered_map<int64_t, int64_t>> ml_comparer(missing_links);
    PriorityComparer<std::unordered_map<int64_t, int64_t>> cl_comparer(candidate_links);
    PriorityComparer<std::vector<int64_t>> default_comparer(priorities);
    PriorityComparer<std::vector<bool>> covered_comparer(covered_nodes);
    Compare cmp;

    switch (kplex_priority) {
    case NodePriority::MAX_UNCOVERED:    
    case NodePriority::MAX_DEGREE:
        cmp = default_comparer.get_greater();
        break;
    
    case NodePriority::MAX_IN_KPLEX: 
        cmp = ml_comparer.get_less(); 
        break;
    
    case NodePriority::MIN_IN_KPLEX: 
        cmp = ml_comparer.get_greater(); 
        break;
    
    case NodePriority::MAX_CANDIDATES:
        cmp = cl_comparer.get_greater();
        break;

    case NodePriority::MIN_CANDIDATES:
        cmp = cl_comparer.get_less();
        break;

    case NodePriority::COMPOSITE:
        default_comparer.set_default(ml_comparer.get_less());
        cl_comparer.set_default(default_comparer.get_less());
        cmp = cl_comparer.get_greater();
        break;

    default:
        cmp = default_comparer.get_less();
        break;
    }

    if (skip_covered) {
        covered_comparer.set_default(cmp);
        cmp = covered_comparer.get_less();
    }

    std::unordered_set<int64_t> candidates;

    for (auto n: neighbors[node]) {
        missing_links[n] = 0;
        candidates.insert(n);
    }

    while (!candidates.empty()) {
        if (kplex_priority == NodePriority::MAX_CANDIDATES
                || kplex_priority == NodePriority::MIN_CANDIDATES
                || kplex_priority == NodePriority::COMPOSITE) {
            candidate_links.clear();
            
            for (auto c: candidates) {
                auto cl = 0;

                for (auto cousin: neighbors[c]) {
                    if (candidates.count(cousin) > 0) {
                        cl++;
                        candidate_links[cousin]++;
                    }
                }

                candidate_links[c] = cl;
            }
        }

        auto min = std::min_element(candidates.begin(), candidates.end(), cmp);
        auto candidate = *min;
        candidates.erase(min);
        kplex.insert(candidate);
        excluded.insert(candidate);
        auto c_neighbors = neighbors[candidate];

        node_callback(candidate);

        // For each node in the k-plex check whether the candidate is its 
        // neighobor. If not, increase its 'missing_links' counter. If its
        // value reaches k, remove all candidates that are not in its 
        // neighborhood.
        for (auto n: kplex) {
            if (!c_neighbors.count(n)) {
                missing_links[n] += 1;

                if (missing_links[n] == k) {
                    for (auto it = candidates.begin(); it != candidates.end();) {
                        if (!neighbors[n].count(*it)) {
                            excluded.insert(*it);
                            it = candidates.erase(it);
                        } else
                            ++it;                        
                    }
                }
            }
        }

        // For each candidate, update the 'missing_links' counter. If it is
        // greater than k, remove it.
        for (auto it = candidates.begin(); it != candidates.end();) {
            if (!c_neighbors.count(*it) && ++missing_links[*it] >= k) {
                excluded.insert(*it);
                it = candidates.erase(it);
            } else
                ++it;
        }

        // Add the neighbors of the new k-plex element to the candidate set, if
        // they have not been already excluded nor are already candidates.
        for (auto n: c_neighbors) {
            if (excluded.count(n) + candidates.count(n) == 0) {
                auto v = (int64_t) kplex.size();
                auto cousins = neighbors[n];

                for (auto c: kplex) 
                    v -= cousins.count(c);

                if (v < k) {
                    missing_links[n] = v;
                    candidates.insert(n);
                } else {
                    excluded.insert(n);
                }
            }
        }
    }

    return kplex;
}

at::Tensor kplex_cover(at::Tensor row, at::Tensor col, int64_t k, int64_t num_nodes,
            NodePriority cover_priority, NodePriority kplex_priority, bool skip_covered = true) {
    std::tie(row, col) = remove_self_loops(row, col);
    std::vector<std::unordered_set<int64_t>> neighbors(num_nodes);
    auto row_acc = row.accessor<int64_t, 1>(), col_acc = col.accessor<int64_t, 1>();
    Compare cover_cmp; 
    std::vector<int64_t> random_p(num_nodes), uncovered_p(num_nodes), degree_p(num_nodes);
    std::vector<bool> covered_nodes(num_nodes);
    std::vector<int64_t> *kplex_priorities;
    PriorityComparer<std::vector<int64_t>> random_cmp(random_p), degree_cmp(degree_p), uncovered_cmp(uncovered_p);

    for (auto i = 0; i < row.size(0); i++) {
        neighbors[row_acc[i]].insert(col_acc[i]);
    }

    for (auto i = 0; i < num_nodes; ++i) {
        uncovered_p[i] = degree_p[i] = (int64_t) neighbors[i].size();
        random_p[i] = i;
    }
    
    std::random_shuffle(random_p.begin(), random_p.end());

    switch (cover_priority) {
    case NodePriority::RANDOM: 
        cover_cmp = random_cmp.get_less(); 
        break;
    
    case NodePriority::MAX_DEGREE: 
        cover_cmp = degree_cmp.get_greater(); 
        break;
    
    case NodePriority::MIN_DEGREE: 
        cover_cmp = degree_cmp.get_less(); 
        break;
    
    case NodePriority::MAX_UNCOVERED: 
        cover_cmp = uncovered_cmp.get_greater(); 
        break;
    
    case NodePriority::MIN_UNCOVERED: 
        cover_cmp = uncovered_cmp.get_less(); 
        break;
    
    default: 
        return at::Tensor();
    }

    switch (kplex_priority) {
    case NodePriority::MAX_DEGREE:     
    case NodePriority::MIN_DEGREE: 
        kplex_priorities = &degree_p; 
        break;
    
    case NodePriority::RANDOM:
        kplex_priorities = &random_p; 
        break;
    
    default:
        kplex_priorities = &uncovered_p; 
        break;
    }

    std::set<int64_t, Compare> candidates(cover_cmp);
    std::vector<std::unordered_set<int64_t>> cover;
    int64_t output_dim = 0;

    for (auto i = 0; i < num_nodes; ++i) {
        candidates.insert(i);
    }

    std::function<void(int64_t)> callback([&](int64_t node) {
        candidates.erase(node);

        if ((cover_priority == NodePriority::MAX_UNCOVERED 
                || cover_priority == NodePriority::MIN_UNCOVERED
                || kplex_priority == NodePriority::MAX_UNCOVERED 
                || kplex_priority == NodePriority::MIN_UNCOVERED
                || kplex_priority == NodePriority::COMPOSITE)
                && !covered_nodes[node]) {
            for (auto cousin: neighbors[node]) {
                if (candidates.count(cousin) > 0) {
                    candidates.erase(cousin);
                    uncovered_p[cousin] -= 1;
                    candidates.insert(cousin);
                }
            }
        }
        
        covered_nodes[node] = true;
    });

    while (!candidates.empty()) {
        auto candidate = *(candidates.begin());
        candidates.erase(candidate);
        auto kplex = find_kplex(neighbors, candidate, k, num_nodes, covered_nodes, 
                                kplex_priority, *kplex_priorities, skip_covered, callback);
        output_dim += kplex.size();
        cover.push_back(kplex);
    }

    auto index = at::zeros({2, output_dim}, row.options());
    auto index_acc = index.accessor<int64_t, 2>();
    auto idx = 0;

    for (size_t cover_id = 0; cover_id < cover.size(); ++cover_id) {
        for (auto node: cover[cover_id]) {
            index_acc[0][idx] = node;
            index_acc[1][idx] = cover_id;
            ++idx;
        }
    }

    return index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kplex_cover", &kplex_cover, "K-plex Cover (CPU)");

    py::enum_<NodePriority>(m, "NodePriority")
        .value("random", NodePriority::RANDOM)
        .value("min_degree", NodePriority::MIN_DEGREE)
        .value("max_degree", NodePriority::MAX_DEGREE)
        .value("min_uncovered", NodePriority::MIN_UNCOVERED)
        .value("max_uncovered", NodePriority::MAX_UNCOVERED)
        .value("min_in_kplex", NodePriority::MIN_IN_KPLEX)
        .value("max_in_kplex", NodePriority::MAX_IN_KPLEX)
        .value("min_candidates", NodePriority::MIN_CANDIDATES)
        .value("max_candidates", NodePriority::MAX_CANDIDATES)
        .value("composite", NodePriority::COMPOSITE)
        .export_values();
}
