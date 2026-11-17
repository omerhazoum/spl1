#ifndef MEMORY_BLOCK_H_
#define MEMORY_BLOCK_H_

#include <string>

enum class IdleAction { KeepCached, Destroy };

class MemoryBlock{
protected:
    size_t size;
    int refCount;
    int* data;
public:
    MemoryBlock(size_t size);
    virtual IdleAction onLastRelease() const = 0;
    virtual MemoryBlock* clone() const = 0;
    virtual ~MemoryBlock();
    int readFromBlock(std::size_t offset) const;
    void writeToBlock(std::size_t offset, int value);
    void acquire();
    IdleAction release();
    int getRefCount() const { return refCount; }
};

class KVBlock: public MemoryBlock{
public:
    KVBlock(size_t size);
    virtual IdleAction onLastRelease() const;
    virtual MemoryBlock* clone() const;
};

class EmbeddingWorkspace: public MemoryBlock{
public:
    EmbeddingWorkspace(size_t size);
    virtual IdleAction onLastRelease() const;
    virtual MemoryBlock* clone() const;
};

#endif