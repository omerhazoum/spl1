#include <fstream>
#include <stdexcept>
#include <map>
#include <utility>
#include <algorithm>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "Request.h"
#include "SchedulingPolicy.h"
#include "GPU.h"
#include "MemoryBlock.h"

using namespace std;
using json = nlohmann::json;

vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const
{
    (void)gpus;
    vector<int> indices;
    for (int i = 0; i < static_cast<int>(eligible.size()); i++)
    {
        indices.push_back(i);
    }
    sort(indices.begin(), indices.end(), [&eligible](int a, int b)
         {
        if (eligible[a]->getAdmissionCycle()!=eligible[b]->getAdmissionCycle()) {
            return eligible[a]->getAdmissionCycle() < eligible[b]->getAdmissionCycle();
        }
        return eligible[a]->getId() < eligible[b]->getId(); });
    return indices;
}

SchedulingPolicy *ArrivalOrderPolicy::clone() const
{
    return new ArrivalOrderPolicy(*this);
}

vector<int> ShortestJobPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const
{
    vector<int> indices;
    for (int i = 0; i < static_cast<int>(eligible.size()); i++)
    {
        indices.push_back(i);
    }
    sort(indices.begin(), indices.end(), [&eligible, &gpus](int a, int b)
         {
        size_t blocksA = eligible[a]->getMaxNeededMemory();
        size_t blocksB = eligible[b]->getMaxNeededMemory();
        if (!gpus.empty()) {
            blocksA = gpus[0].getBlocksForMemory(blocksA);
            blocksB = gpus[0].getBlocksForMemory(blocksB);
        }
        if (blocksA != blocksB) {
            return blocksA < blocksB;
        }
        return eligible[a]->getId() < eligible[b]->getId(); });
    return indices;
}

SchedulingPolicy *ShortestJobPolicy::clone() const
{
    return new ShortestJobPolicy(*this);
}

vector<int> LongestPrefixPolicy::choose(const std::vector<const Request *> &eligible, const std::vector<GPU> &gpus) const
{
    vector<size_t> hValues(eligible.size(), 0);
    vector<int> indices;
    for (int i = 0; i < static_cast<int>(eligible.size()); i++)
    {
        indices.push_back(i);
        vector<string> tokens = eligible[i]->getCacheableTokens();
        if (!tokens.empty())
        {
            const Model &model = eligible[i]->getModel();
            for (const GPU &g : gpus)
            {
                PrefixMatch match = g.findLongestCachedPrefix(
                    model.getId(),
                    model.getEmbeddingSize(),
                    tokens);
                if (hValues[i] < match.matchedTokens)
                {
                    hValues[i] = match.matchedTokens;
                }
            }
        }
    }
     sort(indices.begin(), indices.end(), [&hValues, &eligible](int a, int b)
             {
        if (hValues[a] != hValues[b]) {
            return hValues[a] > hValues[b];
        }
        return eligible[a]->getId() < eligible[b]->getId(); });
        return indices;
}

SchedulingPolicy *LongestPrefixPolicy::clone() const
{
    return new LongestPrefixPolicy(*this);
}