#pragma once

#include "Type.h"

#include <set>
#include <string>
#include <vector>

namespace SM_V1
{
struct ModelIdAssignment
{
    int original_id = 0;
    int runtime_id = 0;
};

// Normalize IDs only on the in-memory model objects before a multi-model
// search.  Source files are never modified.  Existing positive unique IDs
// are preserved; invalid or duplicate IDs receive the smallest unused
// positive integer in deterministic input order.
inline bool normalizeTemplateIds(const std::vector<T_T::Template::Ptr>& models,
                                 std::vector<ModelIdAssignment>& assignments,
                                 std::string* error = nullptr)
{
    assignments.clear();
    std::set<int> used;
    int nextId = 1;
    for (const auto& model : models) {
        if (!model) {
            if (error) *error = "model list contains a null model";
            assignments.clear();
            return false;
        }
        const int originalId = model->template_cfg.id;
        int runtimeId = originalId;
        if (runtimeId <= 0 || used.count(runtimeId) != 0) {
            while (used.count(nextId) != 0) ++nextId;
            runtimeId = nextId++;
            model->template_cfg.id = runtimeId;
        }
        used.insert(runtimeId);
        ModelIdAssignment assignment;
        assignment.original_id = originalId;
        assignment.runtime_id = runtimeId;
        assignments.push_back(assignment);
    }
    return true;
}
}
