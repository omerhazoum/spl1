#include <fstream>
#include <stdexcept>
#include <map>
#include <utility>
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
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), reservedSlots(0), requestReservations(), blockPrefixes(), cacheHits(0), evictions(0) {}

size_t GPU::getCacheHits() const { return cacheHits; }

size_t GPU::getEvictions() const { return evictions; }

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
            blockPrefixes.erase(i);
            evictions++;
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

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot){
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot){
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId){
if (block>=memory.size()||memory[block]==nullptr) return;
    for(const std::pair<const std::size_t, std::pair<std::string, std::vector<std::string>>>& model:blockPrefixes){
        if(model.first!=block && model.second.first==modelId&&model.second.second==prefixTokens){
            return;
        }
    }
blockPrefixes[block] = {modelId,prefixTokens};
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const{
if(embeddingSize==0||blockSize/embeddingSize==0) return {0, {}};
size_t t = blockSize / embeddingSize;
size_t compTokens =0;
vector<std::size_t> blockIds;
for (size_t len = t; len <= promptTokens.size(); len += t){
    vector<string> currentPre(promptTokens.begin(),promptTokens.begin()+len);
    bool found = false;
    for(const pair<const size_t, pair<string, vector<std::string>>>& model:blockPrefixes){
        if (model.second.first == modelId && model.second.second == currentPre){
            blockIds.push_back(model.first);
            compTokens = len;
            found = true;
            break;
        }
    }
    if (!found) break;
}
return {compTokens,blockIds};
}

 void GPU::copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks){
    vector<std::size_t> verifiedSlots;
    for(size_t slot: blocks){
    if(slot>=other.memory.size()||other.memory[slot]==nullptr||other.blockPrefixes.count(slot) == 0) continue;
        const pair<string, vector<string>>& prefixInfo = other.blockPrefixes.at(slot);
        bool isExist = false;
        for (const pair<const size_t, pair<string, vector<string>>>& model:blockPrefixes)
        if ((model.second == prefixInfo)&&(!isExist)){
            verifiedSlots.push_back(model.first);
            isExist = true;
        }
        if (isExist) continue;
        size_t availableSlot=0;
        bool found=false;
        for (size_t i=0;i<memory.size()&&(!found);i++)
        {
            if (memory[i]==nullptr){
                availableSlot=i;
                found = true;
            }
        }
        if (!found){
             for (size_t i=0;i<memory.size()&&(!found);i++)
        {
            if (memory[i]!=nullptr&&memory[i]->getRefCount()==0){
                bool goodForUse=false;
                for(size_t vslot:verifiedSlots){
                    if (vslot==i){
                        goodForUse=true;
                        break;
                    }
                }
                if ((!goodForUse)&&(!found)) {
                    availableSlot = i;
                    delete memory[i];
                    memory[i]=nullptr;
                    blockPrefixes.erase(i);
                    evictions++;
                    found = true;
                }
            }
        }
  }
  if (!found) continue;
  memory[availableSlot] = other.memory[slot]->clone();
  blockPrefixes[availableSlot] = prefixInfo;
  verifiedSlots.push_back(availableSlot);
 }
}

 void GPU::acquire(std::size_t slot){
    if (blockPrefixes.count(slot) > 0) {
        cacheHits++;
    }
    memory[slot]->acquire();
}

 void GPU::release(std::size_t slot){
    IdleAction act = memory[slot]->release();
    if(memory[slot]->getRefCount()==0 && (act==IdleAction::Destroy || blockPrefixes.count(slot)==0)){
        delete memory[slot];
        memory[slot] = nullptr;
        blockPrefixes.erase(slot);
    }
 }

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const{
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value){
    memory[blockId]->writeToBlock(offset, value);
}

//rule of 5 implemention

GPU::~GPU(){
    for(size_t i=0;i<memory.size();i++)
    {
        if(memory[i]!=nullptr)
        {
            delete memory[i];
            memory[i] = nullptr;
        }
    }
}

GPU::GPU(const GPU& other): blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(other.memory.size(), nullptr),
      reservedSlots(other.reservedSlots),
      requestReservations(other.requestReservations),
      blockPrefixes(other.blockPrefixes),
      cacheHits(other.cacheHits),
      evictions(other.evictions)
      {
     for(size_t i=0;i<other.memory.size();i++)
    {
        if(other.memory[i]!=nullptr)
        {
            this->memory[i] = other.memory[i]->clone();
            for (size_t r = 0; r < other.memory[i]->getRefCount(); r++) {
                this->memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other){
     if (this==&other)return *this;
     for (size_t i=0;i<memory.size();i++)
     {
        delete memory[i];
        memory[i]= nullptr;
     }
      blocksNum = (other.blocksNum);
      blockSize = (other.blockSize);
      memory.assign(other.memory.size(), nullptr);
      reservedSlots = (other.reservedSlots);
      requestReservations = (other.requestReservations);
      blockPrefixes = (other.blockPrefixes);
      cacheHits = other.cacheHits;
      evictions = other.evictions;
     for(size_t i=0;i<other.memory.size();i++)
    {
        if(other.memory[i]!=nullptr)
        {
            this->memory[i] = other.memory[i]->clone();
            for (size_t r = 0; r < other.memory[i]->getRefCount(); r++) {
                this->memory[i]->acquire();
            }
        }
    }
    return *this;
}

GPU::GPU(GPU&& other) noexcept
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(move(other.memory)),
      reservedSlots(other.reservedSlots),
      requestReservations(move(other.requestReservations)),
      blockPrefixes(move(other.blockPrefixes)), 
      cacheHits(other.cacheHits),
      evictions(other.evictions){
    other.blocksNum = 0;
    other.blockSize = 0;
    other.reservedSlots = 0;
    other.cacheHits=0;
    other.evictions=0;
}

GPU& GPU::operator=(GPU&& other) noexcept{
      if (this==&other)return *this;
     for (size_t i=0;i<memory.size();i++)
     {
        delete memory[i];
        memory[i]= nullptr;
     }
      blocksNum = (other.blocksNum);
      blockSize = (other.blockSize);
      memory = move(other.memory);
      reservedSlots = (other.reservedSlots);
      requestReservations = move(other.requestReservations);
      blockPrefixes = move(other.blockPrefixes);
      cacheHits = other.cacheHits;
      evictions = other.evictions;
      other.blocksNum = 0;
      other.blockSize = 0;
      other.reservedSlots = 0;
      other.cacheHits=0;
      other.evictions=0;
    return *this;
}

