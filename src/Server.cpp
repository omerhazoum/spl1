#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include <tuple>
#include <iostream>

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}
Server::Server(std::string configPath) {
    this->currentCycle = 0;
    Config JSONRead = readConfig(configPath);
    for(GPUConfig g:JSONRead.gpus)
    {
        std::size_t blocksNum = g.memoryBytes / JSONRead.blockSize;
        GPU gpu(blocksNum,JSONRead.blockSize);
        this->gpus.push_back(gpu);
    }
    for (const ModelConfig& m:JSONRead.models)
    {
        this->models.push_back(new Model(m.id,m.embeddingSize));
    }
    for (const RequestConfig& r:JSONRead.requests)
    {
        const Model* foundModel = nullptr;
        for (const Model* m : this->models) {
            if (m->getId() == r.modelId) {  
            foundModel = m;
            break;
            }
        }
        if (r.type==RequestType::Generative)
        {
            this->nonSubmittedRequests.push_back(new GenerativeRequest(r.id,*foundModel,r.admissionCycle,r.prompt, r.scriptedOutput, r.maxOutputTokens));
        }
        else
        {
            this->nonSubmittedRequests.push_back(new EmbeddingRequest(r.id,*foundModel,r.admissionCycle,r.prompt));
        }
    }
    if (JSONRead.policy == "arrivalOrder" || JSONRead.policy == "ArrivalOrderPolicy") {
        this->policy = new ArrivalOrderPolicy();
    } else if (JSONRead.policy == "shortestJob" || JSONRead.policy == "ShortestJobPolicy") {
        this->policy = new ShortestJobPolicy();
    } else if (JSONRead.policy == "longestPrefix" || JSONRead.policy == "LongestPrefixPolicy") {
        this->policy = new LongestPrefixPolicy();
    }
}
Server::~Server() {
    
    for (Model* m : this->models) {
        delete m;
    }
    for (Request* r : this->nonSubmittedRequests) {
        delete r;
    }
    for (Request* r : this->waitingRequests) {
        delete r;
    }
    for (Request* r : this->activeRequests) {
        delete r;
    }
    delete this->policy;
}
void Server::serve() {
    while (step());
    int totalCacheHits = 0;
    int totalEvictions = 0;

    for (const GPU& gpu : this->gpus) {
        totalCacheHits += gpu.getCacheHits();
        totalEvictions += gpu.getEvictions();
    }

    std::cout << "summary: cycles=" << this->currentCycle
              << " cache_hits=" << totalCacheHits
              << " evictions=" << totalEvictions << std::endl;
}
bool Server::step() {
    if (this->nonSubmittedRequests.empty() && this->waitingRequests.empty() && this->activeRequests.empty()) {
        return false; // No more requests to process
    }

    submitArrivedRequests();
    admitWaitingRequests();
    processActiveRequests();

    currentCycle++;
    return true;
}
void Server::submitArrivedRequests() {
    for (auto it = this->nonSubmittedRequests.begin(); it != this->nonSubmittedRequests.end();) {
        Request* request = *it;
        if (request->getAdmissionCycle() == this->currentCycle) {
            bool canEverFit = false;
            for (const GPU& gpu : this->gpus) {
                if (gpu.getBlocksForMemory(request->getMaxNeededMemory()) <= gpu.getBlocksNum()) {
                    canEverFit = true;
                    break;
                }
            }

            if (!canEverFit) {
                std::cout << "rejected " << request->getId()
                          << " cycle=" << this->currentCycle << std::endl;
                delete request;
            } else {
                this->waitingRequests.push_back(request);
            }
            it = this->nonSubmittedRequests.erase(it);
        } else {
            ++it;
        }
    }
}
void Server::admitWaitingRequests() {
    const std::vector<const Request*> eligibleRequests(this->waitingRequests.begin(), this->waitingRequests.end());
    std::vector<int> chosenIndices = this->policy->choose(eligibleRequests, this->gpus);

    for (int index : chosenIndices) {
        if (tryAdmitRequest(this->waitingRequests[index])) {
            this->waitingRequests[index] = nullptr;
        }
    }

    for (auto it = this->waitingRequests.begin(); it != this->waitingRequests.end();) {
        if ((*it) == nullptr) {
            it = this->waitingRequests.erase(it);
        } else {
            ++it;
        }
    }
}
bool Server::tryAdmitRequest(Request* request) {
    GPU* bestGPU = nullptr;
    size_t bestMatchedTokens = 0;
    size_t bestNeededSlots = 0;

    GPU* longestPrefixGPU = nullptr;
    PrefixMatch longestMatch = {0, {}};

    for (GPU& gpu : this->gpus) {
        PrefixMatch match = gpu.findLongestCachedPrefix(request->getModel().getId(),
            request->getModel().getEmbeddingSize(), request->getCacheableTokens());
        if (match.matchedTokens > longestMatch.matchedTokens) {
            longestPrefixGPU = &gpu;
            longestMatch = match;
        }

        size_t idleCachedBlocks = 0;
        for (size_t blockId : match.blockIds) {
            if (gpu.isIdle(blockId)) {
                ++idleCachedBlocks;
            }
        }

        size_t neededSlots = gpu.getBlocksForMemory(request->getNeededMemory(gpu)) + idleCachedBlocks;
        if (neededSlots > gpu.availableSlots()) {
            continue;
        }

        if (bestGPU == nullptr || match.matchedTokens > bestMatchedTokens ||
            (match.matchedTokens == bestMatchedTokens && neededSlots < bestNeededSlots)) {
            bestGPU = &gpu;
            bestMatchedTokens = match.matchedTokens;
            bestNeededSlots = neededSlots;
        }
    }

    if (bestGPU == nullptr) {
        return false;
    }

    if (longestPrefixGPU != nullptr && longestMatch.matchedTokens > bestMatchedTokens) {
        bestGPU->copyMemoryFromOtherGPU(*longestPrefixGPU, longestMatch.blockIds);
    }

    request->reserveMemory(*bestGPU);
    this->requestToGpu[request->getId()] = bestGPU - this->gpus.data();
    this->activeRequests.push_back(request);
    return true;
}
void Server::processActiveRequests() {
    for (auto it = this->activeRequests.begin(); it != this->activeRequests.end();) {
        Request* request = *it;
        size_t gpuIndex = this->requestToGpu[request->getId()];
        GPU& assignedGPU = this->gpus[gpuIndex];

        request->act(assignedGPU);

        if (request->finished()) {
            std::cout << "finished " << request->getId()
                      << " cycle=" << this->currentCycle
                      << " gpu=" << gpuIndex
                      << ": " << request->getOutput() << std::endl;

            request->deallocateMemory(assignedGPU);
            this->requestToGpu.erase(request->getId());
            delete request;
            it = this->activeRequests.erase(it);
        } else {
            ++it;
        }
    }
}