#include "comparer.hpp"


PriorityComparer::PriorityComparer(std::vector<int64_t>& priority) : priority(priority) {}

Compare PriorityComparer::get_greater() {
    return [this](const int64_t lhs, const int64_t& rhs) {
        return this->priority[lhs] > this->priority[rhs] || (!(this->priority[lhs] < this->priority[rhs]) && lhs < rhs);
    };
}

Compare PriorityComparer::get_less() {
    return [this](const int64_t lhs, const int64_t& rhs) {
        return this->priority[lhs] < this->priority[rhs] || (!(this->priority[lhs] > this->priority[rhs]) && lhs < rhs);
    };
}