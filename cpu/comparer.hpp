#ifndef __COMPARER_HPP__
#define __COMPARER_HPP__

#include <torch/extension.h>


using Compare = std::function<bool(const int64_t&, const int64_t&)>;

template<typename T>
class PriorityComparer {
private:
    T& priority;
    Compare default_compare;

public:
    PriorityComparer(T& priority) : priority(priority) {
        default_compare = [](const int64_t& lhs, const int64_t& rhs){ return lhs < rhs; };
    }

    PriorityComparer(T& priority, const Compare& default_compare) : priority(priority), default_compare(default_compare) {}

    void set_default(const Compare& default_compare) {
        this->default_compare = default_compare;
    }

    Compare get_less() {
        return [this](const int64_t lhs, const int64_t& rhs) {
            return this->priority[lhs] < this->priority[rhs] || (!(this->priority[lhs] > this->priority[rhs]) && this->default_compare(lhs, rhs));
        };
    }
    
    Compare get_greater() {
        return [this](const int64_t lhs, const int64_t& rhs) {
            return this->priority[lhs] > this->priority[rhs] || (!(this->priority[lhs] < this->priority[rhs]) && this->default_compare(lhs, rhs));
        };
    }
};

#endif  // __COMPARER_HPP__
