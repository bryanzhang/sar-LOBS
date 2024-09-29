#! /usr/bin/python3

import torch
import torch.nn as nn
import hessianorpinv_matmul_vector as custom_kernels
import sys

class LobsDnnModel(nn.Module):
    def __init__(self):
        super(LobsDnnModel, self).__init__()
        self.withReLUs = set([])
        self.hessians = []
        self.hpinvs = []
        self.sampleCount = 0

    def resetHessianStats(self):
        for i in range(0, len(self.hessians)):
            self.hessians[i] = None
        for i in range(0, len(self.hpinvs)):
            self.hpinvs[i] = None
        self.sampleCount = 0

# 批量更新海塞矩阵相关统计值
def updateHessianStats(model, inputs):
    model.eval()
    with torch.no_grad():
        model.sampleCount += inputs.size(0)
        for j, (name, layer) in enumerate(model.named_children()):
            #print(f"Layer Name: {name}, Layer: {layer}")
            outputs = layer(inputs)
            if name in model.withReLUs:
                #print(f"Layer name: {name}, with relu!")
                outputs = torch.relu(outputs)
            # 只对FNN的weight剪枝，只计算这部分的hessian
            if not isinstance(layer, nn.Linear):
                inputs = outputs
                model.hessians.append(None)
                model.hpinvs.append(None)
                continue
            if len(model.hessians) <= j:
                model.hessians.append(torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float))
                model.hpinvs.append(None)
            elif model.hessians[j] is None:
                model.hessians[j] = torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float)
            outer_products = torch.bmm(inputs.unsqueeze(2), inputs.unsqueeze(1))
            model.hessians[j] += outer_products.sum(dim=0)
            inputs = outputs

# 计算各个subblock的hessian和伪逆阵
def calcHessiansAndPinvs(model, gama):
    print("Sample count:", model.sampleCount)
    for i, (name, layer) in enumerate(model.named_children()):
        if model.hessians[i] is None:
            continue
        h = model.hessians[i]
        h /= model.sampleCount
        h += gama * torch.eye(h.size(0))
        h *= 2
        print("Layer:", i, "Hessian sub block Shape: ", h.size())
        print("Layer:", i, "Hessian sub block Rank: ", torch.linalg.matrix_rank(h))

        print("Layer:", i, "Constructing sub block hessian Pseudo-inverse...")
        hpinv = torch.linalg.pinv(h)
        model.hpinvs[i] = hpinv

# 从hessian或逆阵中取元素，只有其对角中分布有subblock，其他均为0
def get_element_from_horpinv(subblock, row, col):
    blk_row_idx = row // subblock.size(0)
    blk_col_idx = col // subblock.size(1)
    if blk_row_idx != blk_col_idx:
        return 0.0
    return subblock[(row % subblock.size(0))][(col % subblock.size(1))]

# 基于magnitude选择要剪枝的节点
def prune_fcn_layer(layer, count):
    flat_weight = layer.weight.flatten()
    weight_abs = torch.abs(flat_weight)
    max_val = torch.max(weight_abs)
    values, indices = torch.topk(weight_abs, count, largest=False)
    threshold = torch.max(values)
    min_val = torch.min(values)
    print(f"Pruning {count} weights according to weight abs.Max weight abs={max_val}, min={min_val}, threshold={threshold}")
    return indices

def optimal_brain_surgeon(layer, indices, h_block, hpinv_block):
    flat_weight = layer.weight.flatten()
    #print("Sample weight:", flat_weight[375129])
    prune_mask = torch.zeros(layer.in_features * layer.out_features, dtype=torch.bool)
    prune_mask.index_fill_(0, indices, True)

    flat_weight = flat_weight.unsqueeze(1)
    #print("Sample weight2:", flat_weight[375129][0])

    accum_factor_vector = torch.zeros((layer.in_features * layer.out_features, 1), dtype=torch.float)
    for pos in indices:
        #accum_factor_vector[pos][0] = get_element_from_horpinv(h_block, pos, pos) * flat_weight[pos][0]
       val = get_element_from_horpinv(h_block, pos, pos) * flat_weight[pos][0]
       #print("val: ", val, "pos:", pos, "elem:", get_element_from_horpinv(h_block, pos, pos), "w:", flat_weight[pos][0])
       accum_factor_vector[pos][0] = val
    print(accum_factor_vector)
    original_delta = (-1) * custom_kernels.hessianorpinv_matmul_vector(hpinv_block, accum_factor_vector, layer.out_features)
    w = flat_weight + original_delta
    prune_mask = prune_mask.unsqueeze(1)
    w.masked_fill_(prune_mask, 0.0) # 对剪枝位置进行屏蔽，以规避计算的不精确
    delta_w = w - flat_weight
    loss = torch.mm(torch.transpose(custom_kernels.hessianorpinv_matmul_vector(h_block, delta_w, layer.out_features), 0, 1), delta_w) / 2.0

    flat_weight += delta_w
    return flat_weight.squeeze(1).view(layer.out_features, layer.in_features), loss[0][0].item(), original_delta

def optimal_brain_surgeon_v2(layer, indices, h_block):
    flat_weight = layer.weight.flatten()
    accum_factor_vector = torch.zeros(layer.in_features * layer.out_features, dtype=torch.float)
    mask = torch.zeros(layer.in_features * layer.out_features, dtype=torch.bool)
    mask[indices] = True
    accum_factor_vector.scatter_(0, indices, flat_weight[mask])
    accum_factor_vector = accum_factor_vector.unsqueeze(1)
    original_beta = custom_kernels.hessianorpinv_matmul_vector(h_block, accum_factor_vector, layer.out_features)
    original_epsilon = original_beta
    row_mask = torch.ones(layer.in_features * layer.out_features, dtype=torch.bool)
    row_mask[indices] = False
    epsilon = original_beta[row_mask]
    assert epsilon.size(0) == layer.in_features * layer.out_features - indices.size(0)
    # 求裁剪之后的各个子逆阵
    hessian_mask = row_mask.view(layer.out_features, layer.in_features)
    hpinvs = []
    CHECK_RANK=False
    full_ranks = 0
    for output_no in range(0, layer.out_features):
        subblock = h_block.clone()
        mask = hessian_mask[output_no]
        subblock = subblock[mask]
        subblock = subblock[:, mask]
        #subblock = subblock + 2 * gama * torch.eye(subblock.size(0))
        subblock = subblock.contiguous()
        if CHECK_RANK:
            if torch.linalg.matrix_rank(subblock).item() == subblock.size(0):
                #if subblock.size(0) < 10:
                #    print(subblock)
                full_ranks += 1
        inv = torch.linalg.inv(subblock)
        hpinvs.append(inv)
    if CHECK_RANK:
        print("Full ranks:", full_ranks, "/", layer.out_features)
    epsilon = epsilon.contiguous()
    delta_star = custom_kernels.hessianorpinv_matmul_vector_v1(hpinvs, epsilon)
    delta = torch.zeros(layer.in_features * layer.out_features, dtype=torch.float)
    pre_pos = 0
    compact_pos = 0
    indices, _ = torch.sort(indices)
    for pos in indices:
        #print("Pre&Cur Pos:", pre_pos, pos, file=sys.stderr)
        for i in range(pre_pos, pos):
            delta[i] = delta_star[compact_pos][0]
            compact_pos += 1
        delta[pos] = -flat_weight[pos]
        pre_pos = pos + 1
    for i in range(pre_pos, layer.in_features * layer.out_features):
        delta[i] = delta_star[compact_pos][0]
        compact_pos += 1
    delta = delta.view(layer.out_features * layer.in_features, 1)
    print("Delta:", delta)
    flat_weight = flat_weight.unsqueeze(1)
    print("FlatWeight:", flat_weight)
    flat_weight += delta
    loss = torch.mm(torch.transpose(delta, 0, 1), custom_kernels.hessianorpinv_matmul_vector(h_block, delta, layer.out_features)) / 2.0
    return flat_weight.squeeze(1).view(layer.out_features, layer.in_features), loss[0][0].item(), delta
