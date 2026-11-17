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

void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{gpu.addMemoryBlock(*this, slot);}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{gpu.addMemoryBlock(*this, slot);}