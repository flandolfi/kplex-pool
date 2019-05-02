#ifndef __COMPARER_HPP__
#define __COMPARER_HPP__

#include <torch/extension.h>


using Compare = std::function<bool(const int64_t&, const int64_t&)>;

template<typename T>
class PriorityComparer {
private:
    T& priority;

public:
    PriorityComparer(T& priority) : priority(priority) {};
    
    Compare get_greater() {
        return [this](const int64_t lhs, const int64_t& rhs) {
            return this->priority[lhs] > this->priority[rhs] || (!(this->priority[lhs] < this->priority[rhs]) && lhs < rhs);
        };
    }

    Compare get_less() {
        return [this](const int64_t lhs, const int64_t& rhs) {
            return this->priority[lhs] < this->priority[rhs] || (!(this->priority[lhs] > this->priority[rhs]) && lhs < rhs);
        };
    }
};

#endif  // __COMPARER_HPP__
