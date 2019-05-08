#include <torch/extension.h>

enum class PoolOp {MAX, MIN, MEAN, ADD, MUL, DIV};

std::tuple<at::Tensor, at::Tensor, at::Tensor> 
pool_edges(at::Tensor index_row, at::Tensor index_col, at::Tensor row, at::Tensor col, 
        at::Tensor weight, PoolOp pool_op, int64_t num_nodes) {
    auto row_acc = row.accessor<int64_t, 1>();
    auto col_acc = col.accessor<int64_t, 1>();
    auto idx_row_acc = index_row.accessor<int64_t, 1>();
    auto idx_col_acc = index_col.accessor<int64_t, 1>();
    at::Tensor out_row, out_col, out_weight;

    std::unordered_map<int64_t, std::unordered_set<int64_t>> node_clusters;

    for (auto i = 0; i < index_row.size(0); i++) {
        if (!node_clusters.count(idx_row_acc[i]))
            node_clusters[idx_row_acc[i]] = std::unordered_set<int64_t>({idx_col_acc[i]});
        else 
            node_clusters[idx_row_acc[i]].insert(idx_col_acc[i]);
    }

    auto edge_hash = [=](const std::pair<int64_t, int64_t>& p) {
        auto l = p.first, r = p.second;

        return (size_t) l*num_nodes + r;
    };

    AT_DISPATCH_ALL_TYPES(weight.type(), "pool_edges", [&] {
        auto weight_acc = weight.accessor<scalar_t, 1>();
        std::unordered_map<std::pair<int64_t, int64_t>, scalar_t, decltype(edge_hash)> out_edges(0, edge_hash);
        std::unordered_map<std::pair<int64_t, int64_t>, scalar_t, decltype(edge_hash)> out_edge_count(0, edge_hash);
        std::function<scalar_t(scalar_t, scalar_t)> pool_fun;

        switch (pool_op) {
            case PoolOp::MAX:
                pool_fun = [](scalar_t x, scalar_t y){ return std::max<scalar_t>(x, y); };
                break;
                
            case PoolOp::MIN:
                pool_fun = [](scalar_t x, scalar_t y){ return std::min<scalar_t>(x, y); };
                break;

            case PoolOp::MEAN:
            case PoolOp::ADD:
                pool_fun = [](scalar_t x, scalar_t y){ return x + y; };
                break;

            case PoolOp::MUL:
            case PoolOp::DIV:
                pool_fun = [](scalar_t x, scalar_t y){ return x * y; };
                break;
        }

        for (auto i = 0; i < row.size(0); i++) {
            auto c_from = node_clusters[row_acc[i]], c_to = node_clusters[col_acc[i]];

            for (auto l_node: c_from) {
                for (auto r_node: c_to) {
                    if (!out_edges.count(std::make_pair(l_node, r_node))) {
                        out_edges[std::make_pair(l_node, r_node)] = weight_acc[i];
                        
                        if (pool_op == PoolOp::MEAN)
                            out_edge_count[std::make_pair(l_node, r_node)] = 1;
                    } else {
                        out_edges[std::make_pair(l_node, r_node)] = pool_fun(weight_acc[i], out_edges[std::make_pair(l_node, r_node)]);
                        
                        if (pool_op == PoolOp::MEAN)
                            out_edge_count[std::make_pair(l_node, r_node)]++;
                    }
                }
            }
        }

        auto size = out_edges.size();
        out_row = at::zeros(size, row.options());
        auto out_row_acc = out_row.accessor<int64_t, 1>();
        out_col = at::zeros(size, col.options());
        auto out_col_acc = out_col.accessor<int64_t, 1>();
        out_weight = at::zeros(size, weight.options());
        auto out_weight_acc = out_weight.accessor<scalar_t, 1>();
        auto count = 0;

        for (const auto& map: out_edges) {
            out_row_acc[count] = map.first.first;
            out_col_acc[count] = map.first.second;

            switch (pool_op) {
                case PoolOp::MEAN:
                    out_weight_acc[count] = map.second/out_edge_count[map.first];
                    break;
                
                case PoolOp::DIV:
                    out_weight_acc[count] = 1./map.second;
                    break;

                default:
                    out_weight_acc[count] = map.second;
                    break;
            }

            ++count;
        }
    });

    return {out_row, out_col, out_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pool_edges", &pool_edges, "Pool Edges (CPU)");
    
    py::enum_<PoolOp>(m, "PoolOp")
        .value("max", PoolOp::MAX)    
        .value("min", PoolOp::MIN)
        .value("mean", PoolOp::MEAN)
        .value("add", PoolOp::ADD)
        .value("mul", PoolOp::MUL)
        .value("div", PoolOp::DIV)    
        .export_values();
}
