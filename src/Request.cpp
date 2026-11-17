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
    tokens(model.tokenizer(prompt)), reservedBlocks(0) {}

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
    
}