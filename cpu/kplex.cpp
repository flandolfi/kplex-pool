#include <torch/extension.h>
#include "comparer.hpp"


enum class NodePriority { 
    RANDOM, 
    MIN_DEGREE,
    MAX_DEGREE, 
    MIN_UNCOVERED, 
    MAX_UNCOVERED, 
    MIN_IN_KPLEX,
    MAX_IN_KPLEX 
};

/* From torch_cluster/cpu/utils.h by @rusty1s */
std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row, at::Tensor col) {
    auto mask = row != col;
    return std::make_tuple(row.masked_select(mask), col.masked_select(mask));
}

std::unordered_set<int64_t> find_kplex(const std::vector<std::unordered_set<int64_t>>& neighbors, 
        int64_t node, int64_t k, int64_t num_nodes, NodePriority kplex_priority, const Compare& compare) {
    std::unordered_set<int64_t> excluded({node});
    std::unordered_set<int64_t> kplex({node});
    std::unordered_map<int64_t, int64_t> missing_links({{node, 1}});
    PriorityComparer<std::unordered_map<int64_t, int64_t>> comparer(missing_links);
    Compare cmp = compare;

    switch (kplex_priority) {
    case NodePriority::MAX_IN_KPLEX: 
        cmp = comparer.get_less(); 
        break;
    
    case NodePriority::MIN_IN_KPLEX: 
        cmp = comparer.get_greater(); 
        break;
    
    default:
        break;
    }

    std::set<int64_t, Compare> candidates(cmp);

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

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor> 
kplex_cover(at::Tensor row, at::Tensor col, int64_t k, int64_t num_nodes, bool normalize,
        NodePriority cover_priority, NodePriority kplex_priority, at::Tensor batch) {
    std::tie(row, col) = remove_self_loops(row, col);
    std::vector<std::unordered_set<int64_t>> neighbors(num_nodes);
    auto row_acc = row.accessor<int64_t, 1>(), col_acc = col.accessor<int64_t, 1>();
    Compare cover_cmp, kplex_cmp; 
    std::vector<int64_t> random_p(num_nodes), uncovered_p(num_nodes), degree_p(num_nodes);
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
        return {at::Tensor(), at::Tensor(), -1, -1, at::Tensor()};
    }

    switch (kplex_priority) {
    case NodePriority::RANDOM: 
        kplex_cmp = random_cmp.get_less(); 
        break;
    
    case NodePriority::MAX_DEGREE: 
        kplex_cmp = degree_cmp.get_greater(); 
        break;
    
    case NodePriority::MIN_DEGREE: 
        kplex_cmp = degree_cmp.get_less(); 
        break;
    
    case NodePriority::MAX_UNCOVERED: 
        kplex_cmp = uncovered_cmp.get_greater(); 
        break;
    
    case NodePriority::MIN_UNCOVERED: 
        kplex_cmp = uncovered_cmp.get_less(); 
        break;
    
    default:
        break;
    }

    std::set<int64_t, Compare> candidates(cover_cmp);
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
                        uncovered_p[cousin] -= 1;
                        candidates.insert(cousin);
                    }
                }

                covered_nodes.insert(node);
            }
        }

        cover.push_back(kplex);
    }

    int64_t size = cover.size();
    auto index = at::zeros({2, output_dim}, row.options());
    auto values = at::ones(output_dim, row.options()).toType(at::ScalarType::Float);
    auto out_batch = at::zeros(size, batch.options());
    auto batch_acc = batch.accessor<int64_t, 1>();
    auto out_batch_acc = out_batch.accessor<int64_t, 1>();
    auto index_acc = index.accessor<int64_t, 2>();
    auto insances = at::zeros(output_dim, row.options());
    auto insances_acc = insances.accessor<int64_t, 1>();
    auto idx = 0;

    for (auto cover_id = 0; cover_id < size; ++cover_id) {
        for (auto node: cover[cover_id]) {
            index_acc[0][idx] = node;
            index_acc[1][idx] = cover_id;
            insances_acc[node] += 1;
            out_batch_acc[cover_id] = batch_acc[node];
            ++idx;
        }
    }

    if (normalize) {
        insances = 1. / insances.toType(at::ScalarType::Float);
        values = insances.index_select(0, index[0]);
    }

    return {index, values, num_nodes, size, out_batch};
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
