#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "Request.h"
#include "SchedulingPolicy.h"

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
Server::Server(std::string configPath){
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
       else{
        this->nonSubmittedRequests.push_back(new EmbeddingRequest(r.id,*foundModel,r.admissionCycle,r.prompt));
       }
    }
    if (JSONRead.policy == "arrival_order" || JSONRead.policy == "ArrivalOrderPolicy") {
    this->policy = new ArrivalOrderPolicy();
} else if (JSONRead.policy == "shortest_job" || JSONRead.policy == "ShortestJobPolicy") {
    this->policy = new ShortestJobPolicy();
} else if (JSONRead.policy == "longest_prefix" || JSONRead.policy == "LongestPrefixPolicy") {
    this->policy = new LongestPrefixPolicy();
}
}
Server::~Server() {
   for(Model* m : this->models) {
    delete m;
}
for(Request* r : this->nonSubmittedRequests) {
    delete r;
}
for(Request* r : this->waitingRequests) {
    delete r;
}
for(Request* r : this->activeRequests) {
    delete r;
}
delete this->policy;
}
Server::Server(const Server& other){
 for(const Model* m : other.models) {
    this->models.push_back(new Model(*m));
}
for(const Request* r : other.nonSubmittedRequests) {
   Model* model = nullptr;
    for (Model* m : this->models) {
        if (m->getId() == r->getModel().getId()) {
            model = m;
            break;
        }
    }
        this->nonSubmittedRequests.push_back(r->clone(*model));
}
for(const Request* r : other.waitingRequests) {
     Model* model = nullptr;
    for (Model* m : this->models) {
        if (m->getId() == r->getModel().getId()) {
            model = m;
            break;
        }
    }
        this->waitingRequests.push_back(r->clone(*model));
}
for(const Request* r : other.activeRequests) {
      Model* model = nullptr;
    for (Model* m : this->models) {
        if (m->getId() == r->getModel().getId()) {
            model = m;
            break;
        }
    }
        this->activeRequests.push_back(r->clone(*model));
}
this->policy=other.policy ? other.policy->clone() : nullptr;
this->gpus = other.gpus;
}
Server& Server::operator=(const Server& other) {
    
    if (this != &other) {
        for(Model* m : this->models) {
            delete m;
        }
        for(Request* r : this->nonSubmittedRequests) {
            delete r;
        }
        for(Request* r : this->waitingRequests) {
            delete r;
        }
        for(Request* r : this->activeRequests) {
            delete r;
        }
        delete this->policy;
        this->models.clear();
        this->nonSubmittedRequests.clear();
        this->waitingRequests.clear();
        this->activeRequests.clear();
        for(const Model* m : other.models) {
            this->models.push_back(new Model(*m));
        }
        for(const Request* r : other.nonSubmittedRequests) {
            Model* model = nullptr;
            for (Model* m : this->models) {
                if (m->getId() == r->getModel().getId()) {
                    model = m;
                    break;
                }
            }
            this->nonSubmittedRequests.push_back(r->clone(*model));
        }
        for(const Request* r : other.waitingRequests) {
            Model* model = nullptr;
            for (Model* m : this->models) {
                if (m->getId() == r->getModel().getId()) {
                    model = m;
                    break;
                }
            }
            this->waitingRequests.push_back(r->clone(*model));
        }
        for(const Request* r : other.activeRequests) {
            Model* model = nullptr;
            for (Model* m : this->models) {
                if (m->getId() == r->getModel().getId()) {
                    model = m;
                    break;
                }
            }
            this->activeRequests.push_back(r->clone(*model));
        }
        this->policy = other.policy ? other.policy->clone() : nullptr;
        this->gpus = other.gpus;
    }
    return *this;
}
Server::Server(Server&& other) noexcept {
    this->models = std::move(other.models);
    this->nonSubmittedRequests = std::move(other.nonSubmittedRequests);
    this->waitingRequests = std::move(other.waitingRequests);
    this->activeRequests = std::move(other.activeRequests);
    this->policy = other.policy;
    other.policy = nullptr;
    this->gpus = std::move(other.gpus);
}
Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        for(Model* m : this->models) {
            delete m;
        }
        for(Request* r : this->nonSubmittedRequests) {
            delete r;
        }
        for(Request* r : this->waitingRequests) {
            delete r;
        }
        for(Request* r : this->activeRequests) {
            delete r;
        }
        delete this->policy;

        this->models = std::move(other.models);
        this->nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        this->waitingRequests = std::move(other.waitingRequests);
        this->activeRequests = std::move(other.activeRequests);
        this->policy = other.policy;
        other.policy = nullptr;
        this->gpus = std::move(other.gpus);
    }
    return *this;
}
void Server::serve() {}
bool Server::step() {return false;}