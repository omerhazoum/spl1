#include "MemoryBlock.h"

using namespace std;

MemoryBlock::MemoryBlock(size_t size): size(size), refCount(0), data(nullptr) {
    data = new int[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = 0;
    }
}
MemoryBlock::~MemoryBlock() {
    delete[] data;
}
MemoryBlock::MemoryBlock(const MemoryBlock& other): size(other.size), refCount(0), data(nullptr) {
    data = new int[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = other.data[i];
    }
}
MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        refCount = 0;
        data = new int[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }
    return *this;
}
MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept : size(other.size), refCount(other.refCount), data(other.data) {
    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
}
MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {
    if (this != &other) {
        data = other.data;
        size = other.size;
        refCount = other.refCount;
        other.size = 0;
        other.refCount = 0;
        delete[] data;
        other.data = nullptr;
    }
    return *this;
}
int MemoryBlock::readFromBlock(std::size_t offset) const {
    return data[offset];
}
void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    data[offset] = value;
}
void MemoryBlock::acquire() {
    ++refCount;
}
IdleAction MemoryBlock::release() {
    --refCount;
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}