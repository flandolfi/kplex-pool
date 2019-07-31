#include <torch/extension.h>
#include "disjoint_sets.hpp"


std::tuple<at::Tensor, at::Tensor, at::Tensor>
sort_by_weight(at::Tensor row, at::Tensor col, at::Tensor weight, bool descending = false) {
    at::Tensor perm;
    std::tie(weight, perm) = weight.sort(-1, descending);

    return std::make_tuple(row.index_select(0, perm), col.index_select(0, perm), weight);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> 
simplify_cutoff(at::Tensor row, at::Tensor col, at::Tensor weight, int64_t num_nodes, bool max = true) {
    std::tie(row, col, weight) = sort_by_weight(row, col, weight, max);
    auto row_acc = row.accessor<int64_t, 1>();
    auto col_acc = col.accessor<int64_t, 1>();
    DisjointSets disjoint_sets(num_nodes);
    int64_t i = 0, max_size = 1;

    AT_DISPATCH_ALL_TYPES(weight.type(), "simplify_cutoff", [&] {
        auto weight_acc = weight.accessor<scalar_t, 1>();

        // Add edges to the graph from most important to least important until the graph becomes
        // connected. Continue adding edges if last weight is the same as the next node.
        for (i = 0; i < weight_acc.size(0) && (max_size < num_nodes 
                    || (i > 0 && weight_acc[i] == weight_acc[i - 1])); ++i) {
            auto size = disjoint_sets.merge(row_acc[i], col_acc[i]);
            max_size = std::max(size, max_size);
        }
    });

    return std::make_tuple(row.slice(0, 0, i), col.slice(0, 0, i), weight.slice(0, 0, i));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simplify_cutoff", &simplify_cutoff, "Simplify Graph: Remove Least Important Edges (CPU)");
}
