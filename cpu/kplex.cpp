#include <torch/extension.h>


// The numbering follows this convention:
//  - 0x0-: Ascending order
//  - 0x1-: Descending order
//
// Then:
//  - 0x-0: Degree order
//  - 0x-1: Uncovered neighbors order
//  - 0x-2: Neighbors in k-plex order
//  - 0x-3: Neighbors in candidate set order
//  - 0x-4: Random order
enum class NodePriority : unsigned char { 
    MIN_DEGREE      = 0x00,
    MAX_DEGREE      = 0x10, 
    MIN_UNCOVERED   = 0x01, 
    MAX_UNCOVERED   = 0x11, 
    MAX_IN_KPLEX    = 0x02,
    MIN_IN_KPLEX    = 0x12,
    MIN_CANDIDATES  = 0x03,
    MAX_CANDIDATES  = 0x13,
    RANDOM          = 0x04
};

// We need to know whether two NodePriorities use the same information to sort
// the nodes, hence we distinguish them by their four least significant bits.

struct PriorityHash {
    std::size_t operator()(const NodePriority& p) const { 
        return (unsigned char) p & 0x0F; 
    }
};

struct PriorityEqual {
    bool operator()(const NodePriority& lhs, const NodePriority& rhs) const {
        PriorityHash hash;

        return hash(lhs) == hash(rhs);
    }
};

// We map each NodePriority to a vector of int64_t values, using Hash/Equals
// the two previous structs. This allow us not to generate twice the same
// information.
using PriorityContainer = std::unordered_map<NodePriority, std::vector<int64_t>, PriorityHash, PriorityEqual>;

// Class used to compare two nodes by their priority.
using Compare = std::function<bool(const int64_t&, const int64_t&)>;

// From torch_cluster/cpu/utils.h by @rusty1s
std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row, at::Tensor col) {
    auto mask = row != col;
    return std::make_tuple(row.masked_select(mask), col.masked_select(mask));
}

//                        *** NOTE TO REVIEWERS ***                         
// The following algorithms (FindKPlex and KPlexCover) are just a preliminary
// version and could be highly improved. For example: given that the priority
// values are just numbers from 1 to n, where n is the number of nodes, one
// can use
//    `std::vector<std::unordered_set<int64_t>> candidates(n);`
// and define the candidates as a vector mapping a priority to the nodes
// with that priority, and then extract them by enumerating all the priorities
// from 0 to n, obtaining an amortized complexity of O(n). Nodes that change
// priority during the execution of the algorithm can just be moved to one set
// to another in O(1) or, if their priority becomes higher than the current
// one, can be put in a temporary bucket with highest priority.


// FindKPlex algorithm. Most of the code is needed to perform set operations
// and to manage the priorities and their update. For a more simplified (and
// more understendable) version, see the pseudocode in the article.
std::unordered_set<int64_t> find_kplex(const std::vector<std::unordered_set<int64_t>>& neighbors, int64_t node, int64_t k, 
            Compare kplex_cmp, PriorityContainer& priorities, const std::function<void(int64_t)>& node_callback) {
    std::unordered_set<int64_t> excluded({node});
    std::unordered_set<int64_t> kplex({node});
    std::vector<int64_t>& missing_links = priorities[NodePriority::MAX_IN_KPLEX];
    std::unordered_set<int64_t> candidates;
    missing_links[node] = 1;

    for (auto n: neighbors[node]) {
        missing_links[n] = 0;
        candidates.insert(n);
    }

    while (!candidates.empty()) {
        if (priorities.count(NodePriority::MAX_CANDIDATES)) {
            for (auto c: candidates) {
                priorities[NodePriority::MAX_CANDIDATES][c] = 0;

                for (auto cousin: neighbors[c])
                    if (candidates.count(cousin) > 0) 
                        priorities[NodePriority::MAX_CANDIDATES][c]++;
            }
        }

        // Extract the node with the highest priority. This can deteriorate
        // the efficency of the algorithm.
        auto min = std::min_element(candidates.begin(), candidates.end(), kplex_cmp);
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

// Combine multiple priorities, generating a lexicographic ordering. Ascending
// and descending ordering are defined by the `less` argument.
template<typename T> 
Compare make_comparer(const T& priority, Compare deafault_cmp, bool less = true) {
    if (less)
        return Compare([=, &priority](const int64_t& lhs, const int64_t& rhs) {
            return priority[lhs] < priority[rhs] || (!(priority[lhs] > priority[rhs]) && deafault_cmp(lhs, rhs));
        });

    return Compare([=, &priority](const int64_t& lhs, const int64_t& rhs) {
        return priority[lhs] > priority[rhs] || (!(priority[lhs] < priority[rhs]) && deafault_cmp(lhs, rhs));
    });
}

// Creates the comparer from the list of NodePriority.
Compare build_comparer(const std::vector<std::unordered_set<int64_t>>& neighbors, 
        const std::vector<NodePriority>& priority_types, PriorityContainer& priority_values) {
    Compare cmp([](const int64_t& lhs, const int64_t& rhs){ return lhs < rhs; }); 
    int64_t num_nodes = neighbors.size();

    for (auto p = priority_types.crbegin(); p != priority_types.crend(); ++p) {
        if (!priority_values.count(*p)) {
            priority_values[*p] = std::vector<int64_t>(num_nodes);
            
            switch (*p) {
            case NodePriority::RANDOM:
                for (auto i = 0; i < num_nodes; ++i)
                    priority_values[*p][i] = i;

                std::random_shuffle(priority_values[*p].begin(), priority_values[*p].end());
                break;

            case NodePriority::MAX_DEGREE: 
            case NodePriority::MIN_DEGREE:            
            case NodePriority::MAX_UNCOVERED: 
            case NodePriority::MIN_UNCOVERED: 
                for (auto i = 0; i < num_nodes; ++i) 
                    priority_values[*p][i] = (int64_t) neighbors[i].size();

                break;
            
            default:
                break;
            }
        }
        
        cmp = make_comparer(priority_values[*p], cmp, !((unsigned char) *p & 0xF0));
    }

    return cmp;
}

// KPlexCover algorithm. Most of the code is needed to perform set operations
// and to manage the priorities and their update. For a more simplified (and
// more understendable) version, see the pseudocode in the article.
at::Tensor kplex_cover(at::Tensor row, at::Tensor col, int64_t k, int64_t num_nodes,
            std::vector<NodePriority> cover_priorities, std::vector<NodePriority> kplex_priorities, 
            bool skip_covered = false) {
    std::tie(row, col) = remove_self_loops(row, col);
    std::vector<std::unordered_set<int64_t>> neighbors(num_nodes);
    auto row_acc = row.accessor<int64_t, 1>(), col_acc = col.accessor<int64_t, 1>();
    PriorityContainer priorities;
    std::vector<bool> covered_nodes(num_nodes, false);

    priorities[NodePriority::MAX_IN_KPLEX] = std::vector<int64_t>(num_nodes);

    for (auto i = 0; i < row.size(0); i++)
        neighbors[row_acc[i]].insert(col_acc[i]);

    // Two different comparers: one for KPlexCover, the other for FindKPlex
    Compare cover_cmp = build_comparer(neighbors, cover_priorities, priorities); 
    Compare kplex_cmp = build_comparer(neighbors, kplex_priorities, priorities); 

    // Give highest priority to uncovered nodes.
    if (skip_covered) 
        kplex_cmp = make_comparer(covered_nodes, kplex_cmp);

    // Ordered set. This adds an avoidable  O(log n).
    std::set<int64_t, Compare> candidates(cover_cmp);
    std::vector<std::unordered_set<int64_t>> cover;
    int64_t output_dim = 0;

    for (auto i = 0; i < num_nodes; ++i) {
        candidates.insert(i);
    }

    // Callback used to updata global priorities during the execution of
    // FindKPlex.
    std::function<void(int64_t)> callback([&](int64_t node) {
        candidates.erase(node);

        if (priorities.count(NodePriority::MIN_UNCOVERED) && !covered_nodes[node]) {
            for (auto cousin: neighbors[node]) {
                if (candidates.count(cousin) > 0) {
                    candidates.erase(cousin);
                    priorities[NodePriority::MIN_UNCOVERED][cousin] -= 1;
                    candidates.insert(cousin);
                }
            }
        }
        
        covered_nodes[node] = true;
    });

    // Main loop.
    while (!candidates.empty()) {
        auto candidate = *(candidates.begin());
        candidates.erase(candidates.begin());
        auto kplex = find_kplex(neighbors, candidate, k, kplex_cmp, priorities, callback);
        output_dim += kplex.size();
        cover.push_back(kplex);
    }

    // Generate cover matrix.
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
        .export_values();
}
