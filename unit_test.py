#! /usr/bin/python3

import torch
import torch.nn as nn
import unittest
import lobs_utils

# 使用简单神经网络进行测试
class SimpleDnn(lobs_utils.LobsDnnModel):
    def __init__(self):
        super(SimpleDnn, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2, 1)
        self.withReLUs = set(["fc1"])
        self.fc1.weight.data = torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float64)
        self.fc1.bias.data = torch.tensor([0.0, 0.0], dtype=torch.float64)
        self.fc2.weight.data = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
        self.fc2.bias_data = torch.tensor([0.0], dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TestLOBS(unittest.TestCase):
    def testLOBS_noregularization(self):
        torch.set_printoptions(precision=8)
        model = SimpleDnn()
        model.double()
        model.resetHessianStats()
        inputs = torch.tensor([[1,2,3]], dtype=torch.float64)
        lobs_utils.updateHessianStats(model, inputs)
        inputs = torch.tensor([[-1,0,2]], dtype=torch.float64)
        lobs_utils.updateHessianStats(model, inputs)
        inputs = torch.tensor([[3, -5, 7]], dtype=torch.float64)
        lobs_utils.updateHessianStats(model, inputs)
        layers = list(model.named_children())
        self.assertEquals(layers[0][0], "fc1")
        lobs_utils.calcHessiansAndPinvs(model, 0.0)

        hessian_block = model.hessians[0]
        print("HESSIAN:", hessian_block)
        self.assertEqual(torch.linalg.matrix_rank(hessian_block), 3)
        self.assertTrue(torch.allclose(hessian_block, torch.tensor([[  7.3333,  -8.6667,  14.6667],[ -8.6667,  19.3333, -19.3333],[ 14.6667, -19.3333,  41.3333]], dtype=torch.float64)))
        gbase_block = model.gradients[0]
        hinv_block = model.hpinvs[0]
        self.assertTrue(torch.allclose(torch.mm(hessian_block, hinv_block), torch.eye(3).double(), rtol=1e-5, atol=1e-4))

        layer = layers[0][1]
        prune_seq_2d, loss_table_2d, accum_delta_w_table_2d = lobs_utils.prePrune(model, layer, hessian_block, hinv_block, gbase_block)

        # 第一行的损失及delta weight是否符合预期
        self.assertEqual(prune_seq_2d[0][0], 0)
        self.assertAlmostEqual(loss_table_2d[0][0].item(), 0.00906, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[0][0].double(), torch.tensor([[-0.1],[-0.0176], [0.0273]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        self.assertEqual(prune_seq_2d[0][1], 2)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[0][1].double(), torch.tensor([[-0.1],[0.0552], [0.1]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss_table_2d[0][1].item(), 0.05818139, places=5)

        self.assertEqual(prune_seq_2d[0][2], 1)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[0][2].double(), torch.tensor([[-0.1],[-0.2],[0.1]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss_table_2d[0][2].item(), 0.6294252679246521, places=5)

        # 第二行的损失及delta weight是否符合预期
        self.assertEqual(prune_seq_2d[1][0], 1)
        self.assertAlmostEqual(loss_table_2d[1][0].item(), 0.043787834, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[1][0].double(), torch.tensor([[-0.0848],[-0.1], [-0.0167]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        self.assertEqual(prune_seq_2d[1][1], 0)
        self.assertAlmostEqual(loss_table_2d[1][1].item(), 0.049276660431, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[1][1].double(), torch.tensor([[-0.3],[-0.1], [0.0597]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        self.assertEqual(prune_seq_2d[1][2], 2)
        self.assertAlmostEqual(loss_table_2d[1][2].item(), 1.3936021722356666, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[1][2].double(), torch.tensor([[-0.3],[-0.1], [-0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        original_weight, loss = lobs_utils.greedyPrune(model, layer, 1, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d)
        self.assertTrue(torch.allclose(original_weight, torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertTrue(torch.allclose(layer.weight.data, torch.tensor([[0, 0.1824, -0.0727], [0.3, 0.1, 0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss.item(), 0.00906, places=5)
        layer.weight.data = original_weight

        original_weight, loss = lobs_utils.greedyPrune(model, layer, 3, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d)
        self.assertTrue(torch.allclose(original_weight, torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertTrue(torch.allclose(layer.weight.data, torch.tensor([[0, 0.1824, -0.0727], [0, 0, 0.2597]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss.item(), 0.102124494431, places=5)
        layer.weight.data = original_weight

        original_weight, loss = lobs_utils.greedyPrune(model, layer, 5, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d)
        self.assertTrue(torch.allclose(original_weight, torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertTrue(torch.allclose(layer.weight.data, torch.tensor([[0, 0, 0], [0, 0, 0.2597]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss.item(), 0.7897311523556522, places=5)
        layer.weight.data = original_weight

        original_weight, loss = lobs_utils.greedyPrune(model, layer, 6, prune_seq_2d, loss_table_2d, accum_delta_w_table_2d)
        self.assertTrue(torch.allclose(original_weight, torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.1, 0.2]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertTrue(torch.allclose(layer.weight.data, torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float64), rtol=1e-5, atol=1e-4))
        self.assertAlmostEqual(loss.item(), 2.183333324591319, places=5)
        layer.weight.data = original_weight

    def testLOBS(self):
        return
        model = SimpleDnn()
        model.double()
        model.resetHessianStats()
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.float64)
        lobs_utils.updateHessianStats(model, inputs)
        inputs = torch.tensor([[-1, 0, 2]], dtype=torch.float64)
        lobs_utils.updateHessianStats(model, inputs)
        #inputs = torch.tensor([[-1, 0, 2],[1,2,3]], dtype=torch.float64) # 重复数据不影响结果 
        #lobs_utils.updateHessianStats(model, inputs)
        layers = list(model.named_children())
        self.assertEquals(layers[0][0], "fc1")
        lobs_utils.calcHessiansAndPinvs(model, 10.0)

        hessian_block = model.hessians[0]
        self.assertTrue(torch.equal(hessian_block, torch.tensor([[22, 2, 1], [2, 24, 6], [1, 6, 33]], dtype=torch.float64)))
        gbase_block = model.gradients[0]
        self.assertTrue(torch.equal(gbase_block, torch.tensor([[0], [2], [5]], dtype=torch.float64)))
        hinv_block = model.hpinvs[0]
        self.assertTrue(torch.allclose(hinv_block.double(), torch.tensor([[ 0.0458, -0.0036, -0.0007],[-0.0036,  0.0439, -0.0079],[-0.0007, -0.0079,  0.0318]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        layer = layers[0][1]
        prune_seq_2d, loss_table_2d, accum_delta_w_table_2d = lobs_utils.prePrune(model, layer, hessian_block, hinv_block, gbase_block)
        print("delta:", accum_delta_w_table_2d[0][1])
        self.assertEqual(prune_seq_2d[0][0], 0)
        self.assertAlmostEqual(loss_table_2d[0][0].item(), 0.1091269850730896, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[0][0].double(), torch.tensor([[-0.1],[0.0079], [0.0016]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

        self.assertEqual(prune_seq_2d[0][1], 2)
        print("delta:", accum_delta_w_table_2d[0][1])
        self.assertAlmostEqual(loss_table_2d[0][1].item(), 0.2616666853427887, places=5)
        self.assertTrue(torch.allclose(accum_delta_w_table_2d[0][1].double(), torch.tensor([[-0.1],[-0.0167], [0.10]], dtype=torch.float64), rtol=1e-5, atol=1e-4))

# 正定矩阵每次删除第q行、第q列之后的增量式求逆。
# 根据OBC论文计算
class TestIterativePDInverse(unittest.TestCase):
    def test2x2(self):
        print("2x2")
        block = torch.tensor([[1,2],[2,1]], dtype=torch.float64)
        self.assertTrue(torch.allclose(block, block.t()))
        print("block:", block)
        self.assertEqual(torch.linalg.matrix_rank(block).item(), 2)
        block_inv = torch.linalg.pinv(block)
        print("block_inv:", block_inv)
        self.assertEqual(torch.linalg.matrix_rank(block_inv).item(), 2)
        identity2 = torch.eye(2, dtype=torch.float64)
        res = torch.mm(block, block_inv)
        print("block * inv:", res)
        self.assertTrue(torch.allclose(res, identity2))

        itr = block[1:,1:]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 1)
        print("block_itr:", itr)
        row = block_inv[0].reshape(1, 2)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[0][0].float())[1:,1:]
        print("block_itr inv:", itr_inv)
        identity1 = torch.eye(1, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity1))

    def test3x3(self):
        print("3x3")
        block = torch.tensor([[1,2,4],[2, -1, -5],[4, -5, 3]], dtype=torch.float64)
        self.assertTrue(torch.allclose(block, block.t()))
        print("block:", block)
        self.assertEqual(torch.linalg.matrix_rank(block).item(), 3)
        block_inv = torch.linalg.pinv(block)
        print("block_inv:", block_inv)
        self.assertEqual(torch.linalg.matrix_rank(block_inv).item(), 3)
        identity3 = torch.eye(3, dtype=torch.float64)
        res = torch.mm(block, block_inv)
        print("block * inv:", res)
        self.assertTrue(torch.allclose(res, identity3))

        itr = block[[0,2],:][:,[0,2]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 2)
        print("block_itr:", itr)
        row = block_inv[1].reshape(1, 3)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[1][1].double())[[0,2],:][:,[0,2]]
        print("block_itr inv:", itr_inv)
        identity2 = torch.eye(2, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity2))

    def test6x6(self):
        print("6x6")
        block = torch.tensor([[-2.5, 6, 10, -8, -5, 17], [6,5,3,7,6,4],[10,3,-1,-3,8,6],[-8,7,-3,-2,1,8],[-5,6,8,1,10,-4],[17,4,6,8,-4,9]], dtype=torch.float64)
        self.assertTrue(torch.allclose(block, block.t()))
        print("block:", block)
        self.assertEqual(torch.linalg.matrix_rank(block).item(), 6)
        block_inv = torch.linalg.pinv(block)
        print("block_inv:", block_inv)
        self.assertEqual(torch.linalg.matrix_rank(block_inv).item(), 6)
        identity6 = torch.eye(6, dtype=torch.float64)
        res = torch.mm(block, block_inv)
        print("block * inv:", res)
        self.assertTrue(torch.allclose(res, identity6))

        itr = block[[1,2,3,4,5],:][:,[1,2,3,4,5]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 5)
        print("block_itr:", itr)
        row = block_inv[0].reshape(1, 6)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[0][0].double())[[1,2,3,4,5],:][:,[1,2,3,4,5]]
        print("block_itr inv:", itr_inv)
        identity5 = torch.eye(5, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity5))

        block = itr
        block_inv = itr_inv
        itr = block[[0,1,3,4],:][:,[0,1,3,4]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 4)
        print("block_itr:", itr)
        row = block_inv[2].reshape(1, 5)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[2][2].double())[[0,1,3,4],:][:,[0,1,3,4]]
        print("block_itr inv:", itr_inv)
        identity4 = torch.eye(4, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity4))

        block = itr
        block_inv = itr_inv
        itr = block[[0,1,3],:][:,[0,1,3]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 3)
        print("block_itr:", itr)
        row = block_inv[2].reshape(1, 4)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[2][2].double())[[0,1,3],:][:,[0,1,3]]
        print("block_itr inv:", itr_inv)
        identity3 = torch.eye(3, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity3))

        block = itr
        block_inv = itr_inv
        itr = block[[0,1],:][:,[0,1]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 2)
        print("block_itr:", itr)
        row = block_inv[2].reshape(1, 3)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[2][2].double())[[0,1],:][:,[0,1]]
        print("block_itr inv:", itr_inv)
        identity2 = torch.eye(2, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity2))

        block = itr
        block_inv = itr_inv
        itr = block[[1],:][:,[1]]
        self.assertEqual(torch.linalg.matrix_rank(itr).item(), 1)
        print("block_itr:", itr)
        row = block_inv[0].reshape(1, 2)
        col = torch.transpose(row, 0, 1)
        itr_inv = (block_inv - torch.mm(col, row) / block_inv[0][0].double())[[1],:][:,[1]]
        print("block_itr inv:", itr_inv)
        identity1 = torch.eye(1, dtype=torch.float64)
        res = torch.mm(itr_inv, itr)
        print("block_itr * inv:", res)
        self.assertTrue(torch.allclose(res, identity1))

if __name__ == '__main__':
    unittest.main()

