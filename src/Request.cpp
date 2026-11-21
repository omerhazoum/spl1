#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "Request.h"
#include "GPU.h"
#include "MemoryBlock.h"

using namespace std;
using json = nlohmann::json;

Request::Request(int id, const Model& model, int admissionCycle, string prompt)
    : id(id), model(model), admissionCycle(admissionCycle),
    tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0) {}
Request::Request(const Request& other, const Model& model)
    : id(other.id), model(model), admissionCycle(other.admissionCycle),
    tokens(other.tokens), memoryBlocks(other.memoryBlocks), reservedBlocks(other.reservedBlocks) {}

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
EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest& other, const Model& model)
    : Request(other, model), currentTokenIndex(other.currentTokenIndex), output(other.output) {}

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
    reservedBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    gpu.reserve(reservedBlocks, *this);
    for (size_t i = 0; i < reservedBlocks; ++i) {
        size_t blockId = gpu.allocateBlock(*this);
        memoryBlocks.push_back(blockId);
    }
}
void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }
    size_t offset, scratch;
    if (memoryBlocks.size() == 1) {
        offset = model.getEmbeddingSize();
        scratch = 0;
    }
    else {
        offset = 0;
        scratch = 1;
    }
    model.generateTokenEmbedding({tokens[currentTokenIndex]}, gpu, memoryBlocks[scratch], offset);
    for (size_t i = 0; i < model.getEmbeddingSize(); ++i) {
        int newValue = gpu.readFromBlock(memoryBlocks[scratch], offset + i);
        int ValueSum = gpu.readFromBlock(memoryBlocks[0], i);
        gpu.writeToBlock(memoryBlocks[0], i, ValueSum + newValue);
    }
    ++currentTokenIndex;
    if (finished()) {
        output = "[";
        for (size_t i = 0; i < model.getEmbeddingSize(); ++i) {
            int finalValue = gpu.readFromBlock(memoryBlocks[0], i);
            output += std::to_string(finalValue/(static_cast<int>(currentTokenIndex)));
            if (i < model.getEmbeddingSize() - 1) {
                output += ", ";
            }
        }
        output += "]";
    }
}
Request* EmbeddingRequest::clone(const Model& model) const {
    return new EmbeddingRequest(*this, model);
}

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle,
    string prompt, string scriptedOutput, size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens),
    currentTokenIndex(0), usedBlocks(0), tokenizedOutput(model.tokenizer(scriptedOutput)), currentPrefix() {
        while (tokenizedOutput.size() > maxOutputTokens) {
            tokenizedOutput.pop_back();
        }
    }
GenerativeRequest::GenerativeRequest(const GenerativeRequest& other, const Model& model)
    : Request(other, model), scriptedOutput(other.scriptedOutput), maxOutputTokens(other.maxOutputTokens),
    currentTokenIndex(other.currentTokenIndex), usedBlocks(other.usedBlocks),
    tokenizedOutput(other.tokenizedOutput), currentPrefix(other.currentPrefix) {}
    
size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    size_t matchedTokens = match.matchedTokens;
    return model.getEmbeddingSize()*(tokens.size() - matchedTokens + maxOutputTokens);
}
size_t GenerativeRequest::getMaxNeededMemory() const {
    return model.getEmbeddingSize()*(tokens.size() + maxOutputTokens);
}
vector<string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}
size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks - usedBlocks;
}
void GenerativeRequest::reserveMemory(GPU& gpu) {
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);
    for (size_t slot : match.blockIds) {
        memoryBlocks.push_back(slot);
        gpu.acquire(slot);
    }
    for (size_t i = 0; i < match.matchedTokens; ++i) {
        currentPrefix.push_back(tokens[i]);
    }
    currentTokenIndex = match.matchedTokens;
    reservedBlocks = gpu.getBlocksForMemory(
        model.getEmbeddingSize() * (tokens.size() - match.matchedTokens + maxOutputTokens)
    );
    gpu.reserve(reservedBlocks, *this);
}
bool GenerativeRequest::finished() const {
    return currentTokenIndex >= (tokens.size() + tokenizedOutput.size());
}
void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }
    string currentToken;
    if (currentTokenIndex < tokens.size()) {
        currentToken = tokens[currentTokenIndex];
    }
    else {
        currentToken = tokenizedOutput[currentTokenIndex - tokens.size()];
    }
    currentPrefix.push_back(currentToken);
    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    size_t posInBlock = currentTokenIndex % tokensPerBlock;
    if (posInBlock == 0) {
        size_t newBlockId = gpu.allocateBlock(*this);
        memoryBlocks.push_back(newBlockId);
        ++usedBlocks;
    }
    model.generateTokenEmbedding(currentPrefix, gpu, memoryBlocks.back(), posInBlock * model.getEmbeddingSize());
    ++currentTokenIndex;
    if (currentTokenIndex % tokensPerBlock == 0) {
        gpu.assignPrefixToBlock(memoryBlocks.back(), currentPrefix, model.getId());
    }
}
string GenerativeRequest::getOutput() const {
    string output;
    for (size_t i = 0; i < tokenizedOutput.size(); ++i) {
        if (i > 0) {
            output += " ";
        }
        output += tokenizedOutput[i];
    }
    return output;
}
Request* GenerativeRequest::clone(const Model& model) const {
    return new GenerativeRequest(*this, model);
}