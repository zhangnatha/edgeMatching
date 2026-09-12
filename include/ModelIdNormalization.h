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

// 多模板搜索前仅规范化内存中的模型 ID，不修改源文件。已有的正数唯一 ID 保留；
// 非法或重复 ID 按输入顺序分配尚未使用的最小正整数。
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
