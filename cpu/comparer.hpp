#ifndef __COMPARER_HPP__
#define __COMPARER_HPP__

#include <torch/extension.h>


using Compare = std::function<bool(const int64_t&, const int64_t&)>;

class PriorityComparer {
private:
    std::vector<int64_t>& priority;

public:
    PriorityComparer(std::vector<int64_t>& priority);
    Compare get_greater();
    Compare get_less();
};

#endif  // __COMPARER_HPP__
