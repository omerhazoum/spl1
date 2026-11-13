#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include "GPU.h"
#include "MemoryBlock.h"

using namespace std;
using json = nlohmann::json;

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), reservedSlots(0) {}

 std::size_t GPU::availableSlots() const{
    std::size_t active = 0;
    for (const MemoryBlock* block : memory) {
        if (block != nullptr&& block->getRefCount() > 0) {
            active++;
        }
    }
    return blocksNum - active - reservedSlots;
 }    
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    if (memory == 0|| blockSize == 0) {
        return 0;
    }
    return (memory + blockSize - 1) / blockSize;
}
