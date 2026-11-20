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

Request::Request(int id, const Model& model, int admissionCycle, string prompt)
    : id(id), model(model), admissionCycle(admissionCycle),
    tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0) {}

int Request::getId() const {
    return id;
}
const Model& Request::getModel() const {
    return model;
}
int Request::getAdmissionCycle() const {
    return admissionCycle;
}
void Request::deallocateMemory(GPU& gpu) {
    gpu.unreserve(this->getNeedToUnreserve(), *this);
    for (size_t blockId : memoryBlocks) {
        gpu.release(blockId);
    }
    memoryBlocks.clear();
}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, string prompt)
    : Request(id, model, admissionCycle, prompt), currentTokenIndex(0), output() {}

size_t EmbeddingRequest::getNeededMemory(const GPU&) const {
    return 2*this->getModel().getEmbeddingSize();
}
size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2*this->getModel().getEmbeddingSize();
}
vector<string> EmbeddingRequest::getCacheableTokens() const {
    return {};
}
size_t EmbeddingRequest::getNeedToUnreserve() const {
    return 0;
}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}
bool EmbeddingRequest::finished() const {
    return currentTokenIndex >= tokens.size();
}
string EmbeddingRequest::getOutput() const {
    return output;
}
void EmbeddingRequest::reserveMemory(GPU& gpu) {
    gpu.reserve(this->getMaxNeededMemory(), *this);
}