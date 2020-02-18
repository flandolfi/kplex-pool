#include <torch/extension.h>


// Depth-first search algorithm.
void dfs(int64_t from, int64_t current_component, at::TensorAccessor<int64_t, 1> components, 
         std::vector<std::unordered_set<int64_t>>& neighbors, std::vector<bool>& found) {
    if (found[from])
        return;

    found[from] = true;
    components[from] = current_component;

    for (auto node: neighbors[from]) 
        dfs(node, current_component, components, neighbors, found);
}

// Find the component of each node in the input graph. 
at::Tensor connected_components(at::Tensor row, at::Tensor col, int64_t num_nodes) {
    auto components = at::zeros(num_nodes, row.options());
    std::vector<std::unordered_set<int64_t>> neighbors(num_nodes);
    std::vector<bool> found(num_nodes, false);
    auto row_acc = row.accessor<int64_t, 1>(), col_acc = col.accessor<int64_t, 1>(), 
        components_acc = components.accessor<int64_t, 1>();
    int64_t current_component = 0;

    for (auto i = 0; i < row.size(0); i++) {
        neighbors[row_acc[i]].insert(col_acc[i]);
    }

    for (auto i = 0; i < num_nodes; i++) {
        if (found[i])
            continue;
        
        dfs(i, current_component, components_acc, neighbors, found);
        ++current_component;
    }

    return components;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("connected_components", &connected_components, "Connected Components (CPU)");
}
