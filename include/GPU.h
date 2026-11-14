#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include <map>
#include <utility>

class GenerativeRequest;
class EmbeddingRequest;
class Request;
struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;

class GPU{
public: 
    std::size_t getCacheHits() const;
    std::size_t getEvictions() const;
   ~GPU();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    GPU(GPU&& other) noexcept;
    GPU& operator=(GPU&& other) noexcept;
    GPU(std::size_t blocksNum, std::size_t blockSize);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::size_t reservedSlots;
    std::map<const Request*, std::size_t> requestReservations;
    std::map<std::size_t, std::pair<std::string, std::vector<std::string>>> blockPrefixes;
    std::size_t cacheHits;
    std::size_t evictions;
};

#endif