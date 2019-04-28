#include "disjoint_sets.hpp"


DisjointSets::DisjointSets(int64_t num_nodes) : parent(num_nodes, -1),  size(num_nodes, 1) {};

int64_t DisjointSets::find(int64_t node) {
    auto root = node;

    while (parent[root] > 0)
        root = parent[root];

    return root;
}

int64_t DisjointSets::merge(int64_t l_node, int64_t r_node) {
    auto l_root = find(l_node);
    auto r_root = find(r_node);

    if (l_root == r_root)
        return -1;

    if (size[l_root] < size[r_root])
        std::swap(l_root, r_root);

    parent[r_root] = l_root;
    size[l_root] += size[r_root];

    return size[l_root];
}

int64_t DisjointSets::get_size(int64_t node) {
    auto root = find(node);

    return size[root];
}