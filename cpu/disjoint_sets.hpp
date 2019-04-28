#ifndef __DISJOINT_SETS_HPP__
#define __DISJOINT_SETS_HPP__

#include <torch/extension.h>


class DisjointSets {
private:
    std::vector<int64_t> parent;
    std::vector<int64_t> size;

public:
    DisjointSets(int64_t num_nodes);
    int64_t find(int64_t node);
    int64_t merge(int64_t l_node, int64_t r_node);
    int64_t get_size(int64_t node);
};

#endif  //__DISJOINT_SETS_HPP__
