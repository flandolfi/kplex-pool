#include <torch/extension.h>


enum class NodePriority { 
    RANDOM, 
    MIN_DEGREE,
    MAX_DEGREE, 
    MIN_UNCOVERED, 
    MAX_UNCOVERED, 
    MIN_IN_KPLEX,
    MAX_IN_KPLEX 
};

using Comparator = std::function<bool(const int64_t&, const int64_t&)>;

/* From torch_cluster/cpu/utils.h by @rusty1s */
std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row,
                                                     at::Tensor col) {
    auto mask = row != col;
    return std::make_tuple(row.masked_select(mask), col.masked_select(mask));
}

/* From torch_cluster/cpu/utils.h by @rusty1s */
at::Tensor degree(at::Tensor row, int64_t num_nodes) {
    auto zero = at::zeros(num_nodes, row.options());
    auto one = at::ones(row.size(0), row.options());
    return zero.scatter_add_(0, row, one);
}

std::unordered_set<int64_t> find_kplex(const std::vector<std::unordered_set<int64_t>>& neighbors, 
        int64_t node, int64_t k, int64_t num_nodes, NodePriority kplex_priority, Comparator* comparator = nullptr) {
    std::unordered_set<int64_t> excluded({node});
    std::unordered_set<int64_t> kplex({node});
    std::unordered_map<int64_t, int64_t> missing_links({{node, 1}});
    Comparator& cmp = *comparator;

    switch (kplex_priority) {
    case NodePriority::RANDOM:
    case NodePriority::MAX_DEGREE:
    case NodePriority::MIN_DEGREE:
    case NodePriority::MAX_UNCOVERED:
    case NodePriority::MIN_UNCOVERED:
        break;

    case NodePriority::MAX_IN_KPLEX:
        cmp = [&](const int64_t& l, const int64_t& r){ 
            return missing_links[l] > missing_links[r] || (!(missing_links[l] < missing_links[r]) && l < r); 
        };
        break;

    case NodePriority::MIN_IN_KPLEX:
        cmp = [&](const int64_t& l, const int64_t& r){ 
            return missing_links[l] < missing_links[r] || (!(missing_links[l] > missing_links[r]) && l < r); 
        };
        break;
    }

    std::set<int64_t, Comparator> candidates(cmp);

    for (auto n: neighbors[node]) {
        missing_links[n] = 0;
        candidates.insert(n);
    }

    while (!candidates.empty()) {
        auto candidate = *(candidates.begin());
        candidates.erase(candidates.begin());
        kplex.insert(candidate);
        excluded.insert(candidate);
        auto c_neighbors = neighbors[candidate];

        // For each node in the k-plex check whether the candidate is its 
        // neighobor. If not, increase its 'missing_links' counter. If its
        // value reaches k, remove all candidates that are not in its 
        // neighborhood.
        for (auto n: kplex) {
            if (!c_neighbors.count(n)) {
                missing_links[n] += 1;

                if (missing_links[n] == k) {
                    for (auto c: candidates) {
                        if (!neighbors[n].count(c)) {
                            excluded.insert(c);
                            candidates.erase(c);
                        }
                    }
                }
            }
        }

        // For each candidate, update the 'missing_links' counter. If it is
        // greater than k, remove it.
        auto n = candidates.begin();
        std::vector<int64_t> to_replace;

        while (n != candidates.end()) {
            if (!c_neighbors.count(*n)) {
                auto v = missing_links[*n] + 1;

                if (v >= k) {
                    excluded.insert(*n);
                } else {
                    to_replace.push_back(*n);
                }
                
                auto old_node = *n;
                n = candidates.erase(n);
                missing_links[old_node] = v;
            } else 
                ++n;
        }

        candidates.insert(to_replace.begin(), to_replace.end());

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

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> 
kplex_cover(at::Tensor row, at::Tensor col, int64_t k, int64_t num_nodes, bool normalize,
        NodePriority cover_priority, NodePriority kplex_priority) {
    std::tie(row, col) = remove_self_loops(row, col);
    std::vector<std::unordered_set<int64_t>> neighbors(num_nodes);
    auto deegrees = degree(row, num_nodes);
    auto row_acc = row.accessor<int64_t, 1>(), col_acc = col.accessor<int64_t, 1>(), 
        degree_acc = deegrees.accessor<int64_t, 1>();
    Comparator cover_cmp, *kplex_cmp = nullptr;
    std::vector<int64_t> priorities(num_nodes);

    for (auto i = 0; i < row.size(0); i++) {
        neighbors[row_acc[i]].insert(col_acc[i]);
    }

    switch (cover_priority) {
    case NodePriority::RANDOM:
        for (auto i = 0; i < num_nodes; ++i)
            priorities[i] = i;
        
        std::random_shuffle(priorities.begin(), priorities.end());
        break;

    case NodePriority::MAX_DEGREE:
    case NodePriority::MIN_DEGREE:
    case NodePriority::MAX_UNCOVERED:
    case NodePriority::MIN_UNCOVERED:
        for (auto i = 0; i < num_nodes; ++i)
            priorities[i] = degree_acc[i];

        break;
    
    default:
        return {at::Tensor(), at::Tensor(), -1, -1};
    }

    switch (cover_priority) {
    case NodePriority::RANDOM:
    case NodePriority::MIN_DEGREE:
    case NodePriority::MIN_UNCOVERED:
        cover_cmp = [&](const int64_t& l, const int64_t& r){ 
            return priorities[l] < priorities[r] || (!(priorities[l] > priorities[r]) && l < r); 
        };
        break;

    case NodePriority::MAX_DEGREE:
    case NodePriority::MAX_UNCOVERED:
        cover_cmp = [&](const int64_t& l, const int64_t& r){ 
            return priorities[l] > priorities[r] || (!(priorities[l] < priorities[r]) && l < r); 
        };
        break;
    
    default:
        break;
    }

    switch (kplex_priority) {
    case NodePriority::RANDOM:
    case NodePriority::MAX_DEGREE:
    case NodePriority::MIN_DEGREE:
    case NodePriority::MAX_UNCOVERED:
    case NodePriority::MIN_UNCOVERED:
        kplex_cmp = &cover_cmp;

    default:
        break;
    }

    std::set<int64_t, Comparator> candidates(cover_cmp);
    std::vector<std::unordered_set<int64_t>> cover;
    std::unordered_set<int64_t> covered_nodes;
    int64_t output_dim = 0;

    for (auto i = 0; i < num_nodes; ++i) {
        candidates.insert(i);
    }

    while (!candidates.empty()) {
        auto candidate = *(candidates.begin());
        candidates.erase(candidate);

        auto kplex = find_kplex(neighbors, candidate, k, num_nodes, kplex_priority, kplex_cmp);
        output_dim += kplex.size();

        for (auto node: kplex) {
            candidates.erase(node);

            if ((cover_priority == NodePriority::MAX_UNCOVERED || cover_priority == NodePriority::MIN_UNCOVERED) 
                    && covered_nodes.count(node) == 0) {
                for (auto cousin: neighbors[node]) {
                    if (candidates.count(cousin) > 0) {
                        candidates.erase(cousin);
                        priorities[cousin] -= 1;
                        candidates.insert(cousin);
                    }
                }

                covered_nodes.insert(node);
            }
        }

        cover.push_back(kplex);
    }

    auto index = at::zeros({2, output_dim}, row.options());
    auto values = at::ones(output_dim, row.options()).toType(at::ScalarType::Float);
    auto index_acc = index.accessor<int64_t, 2>();
    int64_t size = cover.size();
    auto idx = 0;

    for (auto cover_id = 0; cover_id < size; ++cover_id) {
        for (auto node: cover[cover_id]) {
            index_acc[0][idx] = node;
            index_acc[1][idx] = cover_id;
            ++idx;
        }
    }

    if (normalize) {
        auto clusters_per_node = degree(index[0], num_nodes).toType(at::ScalarType::Float);
        values = 1. / clusters_per_node.index_select(0, index[0]);
    }

    return {index, values, num_nodes, size};
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
        .export_values();
}
