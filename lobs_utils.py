#! /usr/bin/python3

import torch
import torch.nn as nn
import hessianorpinv_matmul_vector as custom_kernels
import sys
import heapq

class LobsDnnModel(nn.Module):
    def __init__(self):
        super(LobsDnnModel, self).__init__()
        self.withReLUs = set([])
        self.hessians = []
        self.hpinvs = []
        self.hbases = []
        self.gradients = []
        self.sampleCount = 0
        self.alpha = 0.0 # LOBS L2正则化系数

    def resetHessianStats(self):
        for i in range(0, len(self.hessians)):
            self.hessians[i] = None
            self.hpinvs[i] = None
            self.hbases[i] = None
            self.gradients[i] = None
        self.sampleCount = 0

# 批量更新海塞矩阵相关统计值
def updateHessianStats(model, inputs):
    model.eval()
    with torch.no_grad():
        model.sampleCount += inputs.size(0)
        for j, (name, layer) in enumerate(model.named_children()):
            outputs = layer(inputs)
            if name in model.withReLUs:
                outputs = torch.relu(outputs)
            # 只对FNN的weight剪枝，只计算这部分的hessian
            if not isinstance(layer, nn.Linear):
                inputs = outputs
                model.hessians.append(None)
                model.hpinvs.append(None)
                model.hbases.append(None)
                model.gradients.append(None)
                continue
            if len(model.hessians) <= j:
                model.hessians.append(torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float64))
                model.hbases.append(torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float64))
                model.hpinvs.append(None)
                model.gradients.append(torch.zeros((inputs.size(1), 1), dtype=torch.float64))
            elif model.hessians[j] is None:
                model.hessians[j] = torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float64)
                model.hbases[j] = torch.zeros((inputs.size(1), inputs.size(1)), dtype=torch.float64)
                model.gradients[j] = torch.zeros((inputs.size(1), 1), dtype=torch.float64)
            outer_products = torch.bmm(inputs.unsqueeze(2), inputs.unsqueeze(1))
            model.hessians[j] += outer_products.sum(dim=0)
            dim=inputs.size(1)
            model.gradients[j] += inputs.sum(dim=0).view(dim, 1)
            inputs = outputs

# 计算各个subblock的hessian和伪逆阵
def calcHessiansAndPinvs(model, alpha):
    print("Sample count:", model.sampleCount)
    model.alpha = alpha
    for i, (name, layer) in enumerate(model.named_children()):
        if model.hessians[i] is None:
            continue
        h = model.hessians[i]
        h /= model.sampleCount
        h *= 2
        hbase = h.clone()
        h += 2 * alpha * torch.eye(h.size(0))
        print("Layer:", i, "Hessian sub block Shape: ", h.size())
        print("Layer:", i, "Hessian sub block Rank: ", torch.linalg.matrix_rank(h))

        print("Layer:", i, "Constructing sub block hessian Pseudo-inverse...")
        hpinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
        #hpinv = torch.linalg.pinv(h)
        model.hbases[i] = hbase
        model.hpinvs[i] = hpinv
        model.gradients[i] *= 2
        model.gradients[i] /= model.sampleCount

def prePrune(model, layer, h, hinv, gbase):
    # 逐个输出节点地计算
    gbase = h
    rows = layer.weight.data.size(0)
    cols = layer.weight.data.size(1)
    alpha = model.alpha
    prune_seq_2d = []
    loss_table_2d = []
    accum_delta_w_table_2d = []
    original_h = h
    original_hinv = hinv
    original_gbase = gbase
    for j in range(0, rows):
        print("ROW:", j)
        weight = layer.weight.data[j].clone().view(cols, 1)
        accum_delta_w = torch.zeros(cols, 1, dtype=torch.float64)
        accum_delta_w_itr = torch.zeros(cols, 1, dtype=torch.float64)
        mask = torch.zeros(cols, 1, dtype=torch.bool)
        inverted_mask = torch.ones(cols, 1, dtype=torch.bool)
        retain_indices = torch.nonzero(inverted_mask)
        prune_seq = []
        loss_table = []
        accum_delta_w_table = [] # 可以减少recomputation，但空间大小为rows*rows*cols float，占800M+
        h = original_h
        h_fullcol = original_h
        hinv = original_hinv
        gbase = original_gbase
        for c in range(0, cols):
            min_loss = float('inf')
            min_pos = -1
            min_global_pos = -1
            min_delta_w = None
            min_delta_w_itr = None
            min_itr_inv = None
            for k in range(0, h.size(0)):
                global_pos = retain_indices[k][0]
                inv_row = hinv[:,k].view(1, hinv.size(0))
                decr = torch.mm(torch.transpose(inv_row, 0, 1), inv_row) / hinv[k][k]
                tmp_itr_inv = hinv - decr
                tmp_itr_inv = torch.cat((tmp_itr_inv[:k,:], tmp_itr_inv[(k+1):,:]), dim=0)
                tmp_itr_inv = torch.cat((tmp_itr_inv[:,:k], tmp_itr_inv[:,(k+1):]), dim=1)
                g = torch.mm(h_fullcol, accum_delta_w)
                beta = weight[global_pos][0] * h[:,k].view(h.size(0), 1)
                beta -= g
                beta = torch.cat((beta[:k,], beta[(k+1):,]), dim=0)
                tmp_delta_w = torch.mm(tmp_itr_inv, beta)
                tmp_delta_w = torch.cat((tmp_delta_w[:k,], (-weight[global_pos]).view(1,1), tmp_delta_w[k:,]), dim=0)
                loss = torch.mm(g.transpose(0, 1), tmp_delta_w) + torch.mm(tmp_delta_w.transpose(0, 1), torch.mm(h, tmp_delta_w)) / 2.0
                loss = loss[0].double()
                if loss < min_loss:
                    min_loss = loss
                    min_pos = k
                    min_global_pos = global_pos
                    min_delta_w = tmp_delta_w
                    min_itr_inv = tmp_itr_inv
                if j == 0 and c == 1 and global_pos == 2:
                    print("Prev_LOSS:", loss_table[0])
                    print("MIN_LOSS:", min_loss)
                    print("Prev_DELTA_W:", accum_delta_w_table[0])
                    print("MIN_DELTA_W:", min_delta_w)
                    input("按任意键继续。。。")
            flatten_delta_w = torch.zeros((cols, 1), dtype=torch.float64)
            flatten_delta_w.masked_scatter_(inverted_mask, min_delta_w)
            weight += flatten_delta_w
            accum_delta_w += flatten_delta_w
            accum_delta_w_itr += min_delta_w
            accum_delta_w_itr = torch.cat((accum_delta_w_itr[:min_pos,], accum_delta_w_itr[(min_pos+1):,:]), dim=0)
            hinv = min_itr_inv
            h = torch.cat((h[:min_pos,:], h[(min_pos+1):,:]), dim=0)
            h = torch.cat((h[:,:min_pos], h[:,(min_pos+1):]), dim=1)
            h_fullcol = torch.cat((h_fullcol[:min_pos,:], h_fullcol[(min_pos+1):,:]), dim=0)
            gbase = torch.cat((g[:min_pos,:], g[(min_pos+1):,:]), dim=0)
            mask[min_global_pos][0] = True
            inverted_mask[min_global_pos][0] = False
            retain_indices = torch.nonzero(inverted_mask)
            prune_seq.append(min_global_pos)
            loss_table.append(min_loss)
            accum_delta_w_table.append(accum_delta_w.clone())
        prune_seq_2d.append(prune_seq)
        loss_table_2d.append(loss_table)
        accum_delta_w_table_2d.append(accum_delta_w_table)
    return prune_seq_2d, loss_table_2d, accum_delta_w_table_2d

def greedyPrune(model, layer, prune_num, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d):
    rows = layer.weight.data.size(0)
    cols = layer.weight.data.size(1)
    prune_num_rows = [0 for _ in range(rows)]
    heap = []
    accum_loss = 0.0
    original_weight = layer.weight.data.clone()
    for i in range(rows):
        heapq.heappush(heap, (loss_table_2d[i][prune_num_rows[i]], i))
    for _ in range(prune_num):
        loss, row_idx = heapq.heappop(heap)
        accum_loss += loss
        prune_num_rows[row_idx] += 1
        n = prune_num_rows[row_idx]
        if n < cols:
            heapq.heappush(heap, (loss_table_2d[row_idx][n], row_idx))
    for i in range(rows):
        j = prune_num_rows[i] - 1
        if j < 0:
            continue
        layer.weight.data[i] += accum_delta_w_table_2d[i][j].view(cols)
    return original_weight, accum_loss
