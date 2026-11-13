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

 void GPU::reserve(std::size_t n, const Request& request) {
    requestReservations[&request] += n;
    reservedSlots += n;
 }

 void GPU::unreserve(std::size_t n, const Request& request) {
    if (requestReservations.count(&request) == 0) {
        return;
    }
    else if (requestReservations[&request] <= n)
    {
        reservedSlots -= requestReservations[&request];
        requestReservations.erase(&request);
    } else{
    requestReservations[&request] -= n;
    reservedSlots -= n;
    }
}

std::size_t GPU::allocateBlock(const Request& request){
    if(requestReservations.count(&request) == 0||requestReservations[&request] == 0){
        throw std::runtime_error("missing key or invalid reservations number");
        //check whether needed or not!!!
    }
    std::size_t firstSlotEmpty;
    bool found = false;
    for(std::size_t i=0;i<memory.size()&&(!found);i++)
    {
        if(memory[i]==nullptr)
        {
            firstSlotEmpty=i;
            found = true;
        }
    }
     for(std::size_t i=0;i<memory.size()&&(!found);i++)
    {
        if(memory[i] != nullptr && memory[i]->getRefCount() == 0)
        {
            firstSlotEmpty=i;
            delete memory[i];
            memory[i]=nullptr;
            found = true;
        }
    }
    if(!found){
        throw std::runtime_error("no suitable slot available");
        //check whether needed or not!!!
    }
    request.allocateOneBlock(*this,firstSlotEmpty);
    acquire(firstSlotEmpty);
    unreserve(1, request);
    return firstSlotEmpty;
}