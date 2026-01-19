import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

# 计算两组特征向量之间的欧几里得距离
def EculideanDistances(a, b):
    # 对特征向量进行L2归一化
    a = F.normalize(a, p=2, dim=1)  
    b = F.normalize(b, p=2, dim=1)
    # 计算平方和
    sq_a = a**2
    sq_b = b**2
    sq_a_sum = torch.sum(sq_a, dim=1).unsqueeze(1)  # 每个向量自身的平方和
    sq_b_sum = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    # 根据欧几里得距离公式:sqrt((a-b)^2)=sqrt(a^2+b^2-2ab)计算距离矩阵
    return torch.sqrt(sq_a_sum + sq_b_sum - 2*a.mm(bt))

def Inf_Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

# 基于历史代理的记忆三元组学习模块
class MemoryTripletK_reuse(nn.Module):
    def __init__(self, num_classes, DIM):
        super(MemoryTripletK_reuse, self).__init__()
        self.margin = 0.3  # 三元组损失的边界值
        self.noise_rate = 0.75  # 噪声比率
        self.alpha = 1.0  # 混合参数
        self.K = 5  # K近邻数量
        self.iteration = 0  # 当前迭代次数
        self.epoch = 0  # 当前训练轮次
        self.start_check_noise_iteration = 0  # 开始检查噪声的迭代次数
        self.class_num = num_classes  # 类别数量
        self.Weight = torch.ones(DIM,).cuda()  # 样本权重,维度为目标域样本数量

    def forward(self, inputs_col, targets_col, idx_t, inputs_row_t, target_row_t, weight, image_t, model, p_t):
        """
        inputs_col : target_feature  当前批次目标域样本的特征向量 （batch_size, DIM）
        targets_col : predict 当前批次目标域样本的knn预测标签 (batch_size,)
        idx_t : index target 当前批次目标域样本的索引 (batch_size,)
        inputs_row_t : mem_fea_target 历史所有目标域样本的归一化特征 (mem_num, DIM)
        target_row_t : mem_cls_target  历史所有目标域样本的softmax输出分布 (mem_num, num_classes)
        weight : weight 当前批次目标域样本的权重 (batch_size,)
        image_t : x_t 当前批次目标域样本 (batch_size, 3, 224, 224)
        model : model 模型 (ImageClassifier)
        p_t : softmax_out 当前批次目标域样本的softmax输出分布 (batch_size, num_classes)
        
        """
        # 处理目标标签
        _, l = torch.sort(target_row_t, descending=True, dim=1)
        target_row_t = l[0:, 0]
        n = inputs_col.size(0)
        W_t = torch.ones(n, ).cuda()  # 初始化目标域样本权重
        W_tt = torch.ones(n, ).cuda()  # 存储每个样本的可信度得分

        epsilon = 1e-5  # 数值稳定性

        # 初始化损失
        loss = torch.tensor(0.).cuda()  # 三元组损失
        mix_loss = torch.tensor(0.).cuda()  # 混合样本损失
        len = 0

        # 计算特征向量间的欧式距离
        sim_mat_tt = EculideanDistances(inputs_col, inputs_row_t)
        simratio_score_tt = []
        # 设置相同样本间的距离为较大值
        for i in range(n):
            sim_mat_tt[i, idx_t[i]] = 10.  
            
        # 计算每个样本的可信度得分
        for i in range(n):
            t_label = targets_col[i]
            # 获取同类样本的掩码和距离
            nln_mask_tt = (target_row_t == t_label)
            nln_sim_all_tt = sim_mat_tt[i][nln_mask_tt]
            k = min(self.K, nln_sim_all_tt.size(0))
            # 选取K个最近的同类样本
            nln_sim_r_tt = torch.narrow(torch.sort(nln_sim_all_tt, descending=False)[0], 0, 0, k)
            
            # 获取异类样本的掩码和距离
            nun_mask_tt = (target_row_t != t_label)
            nun_sim_all_tt = sim_mat_tt[i][nun_mask_tt]
            k = min(self.K, nun_sim_all_tt.size(0))
            # 选取K个最近的异类样本
            nun_sim_r_tt = torch.narrow(torch.sort(nun_sim_all_tt, descending=False)[0], 0, 0, k)
            
            # 计算可信度得分:异类距离和/同类距离和
            conf_score_tt = (1.0 * torch.sum(nun_sim_r_tt) / torch.sum(nln_sim_r_tt)).item()
            simratio_score_tt.append(conf_score_tt)
            W_tt[i] = conf_score_tt
            W_t[i] = conf_score_tt

        # sort_ranking_score_tt, ind_tgt_tt = torch.sort(torch.tensor(simratio_score_tt), descending=True)
        # _, ind_tgt_tt_low = torch.sort(torch.tensor(simratio_score_tt))
        # top_n_tgt_ind_tt = torch.narrow(ind_tgt_tt, 0, 0, int(self.noise_rate * n))
        # len = torch.mean(torch.tensor(simratio_score_tt)[top_n_tgt_ind_tt])
        # low_n_tgt_ind_tt = torch.narrow(ind_tgt_tt_low, 0, 0, n - int(self.noise_rate * n))
        # W_tt[top_n_tgt_ind_tt] = 1.
        # W_tt[low_n_tgt_ind_tt] = W_tt[low_n_tgt_ind_tt] ** 2 / torch.sum(W_tt[low_n_tgt_ind_tt] ** 2)

        criterion = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None,
                                               reduction='mean') # 定义三元组损失函数
        sum_weight = epsilon
        flag = torch.ones(n,)  # 标记可靠样本
        
        # 对每个样本计算三元组损失
        for i in range(n):
            t_label = targets_col[i]
            # 计算同类样本的权重阈值
            nln_mask_tt = (target_row_t == t_label)
            yuzhi_p = torch.mean(self.Weight[nln_mask_tt])
            # 判断当前样本是否可靠
            if W_t[i] < yuzhi_p:
                flag[i] = 0.
            # 根据权重阈值筛选同类样本    
            nln_mask_tt = nln_mask_tt & (self.Weight >= yuzhi_p)
            nln_sim_all_tt = sim_mat_tt[i][nln_mask_tt]
            k_p = min(self.K, nln_sim_all_tt.size(0))
            # 选择K个最相似的同类样本
            nln_sim_r_tt_idx = torch.narrow(torch.sort(nln_sim_all_tt, descending=True)[1], 0, 0, k_p)
            
            # 计算异类样本的权重阈值
            nun_mask_tt = (target_row_t != t_label)
            yuzhi_n = torch.mean(self.Weight[nun_mask_tt])
            # 根据权重阈值筛选异类样本
            nun_mask_tt = nun_mask_tt & (self.Weight >= yuzhi_n)
            nun_sim_all_tt = sim_mat_tt[i][nun_mask_tt]
            k_n = min(self.K, nun_sim_all_tt.size(0))
            # 选择K个最不相似的异类样本
            nun_sim_r_tt_idx = torch.narrow(torch.sort(nun_sim_all_tt, descending=False)[1], 0, 0, k_n)
            
            # 计算三元组损失
            if k_p == k_n == self.K:
                anchor = inputs_col[i].expand(k_p, inputs_col[i].size(0))  # 复制锚样本
                loss = loss + criterion(anchor, 
                                     (inputs_row_t[nln_mask_tt])[nln_sim_r_tt_idx],  # 正样本
                                     (inputs_row_t[nun_mask_tt])[nun_sim_r_tt_idx]   # 负样本
                                     ) * weight[i]
                sum_weight = sum_weight + weight[i].item()
                
        # 标记可靠样本用于混合
        idx_mix = (flag == 1)
        self.iteration += 1
        loss = loss / sum_weight  # 计算加权平均损失
        self.Weight[idx_t] = W_t  # 更新样本权重
        # reuse
        image_t = image_t[idx_mix]
        if image_t.size(0) <= 1:
            # res_lu = torch.tensor(0.).cuda()
            # res_hu = torch.tensor(0.).cuda()
            return loss, mix_loss, W_tt
        #信息熵检验
        # res_lu = Inf_Entropy(p_t[idx_mix])
        # res_hu = Inf_Entropy(p_t[~idx_mix])
        # res_hu = torch.mean(res_hu)
        # res_lu = torch.mean(res_lu)
        np.random.seed(seed=1)
        np.random.RandomState(seed=1)  #mixup
        len_mix = image_t.size(0)
        t_batch = targets_col[idx_mix]
        lam = (torch.from_numpy(np.random.beta(self.alpha, self.alpha, [len_mix]))).float().cuda()#len(image_t)
        t_batch = torch.eye(self.class_num, device=t_batch.device)[t_batch].cuda()
        #t_batch = torch.eye(self.class_num)[t_batch].cuda()
        shuffle_idx = torch.from_numpy(np.random.permutation(len_mix).astype('int64'))
        mixed_x = (lam * image_t.permute(1, 2, 3, 0) + (1 - lam) * image_t[shuffle_idx].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        mixed_t = (lam * t_batch.permute(1, 0) + (1 - lam) * t_batch[shuffle_idx].permute(1, 0)).permute(1, 0)
        mixed_x, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_t))
        mixed_outputs, _ = model(mixed_x)
        softmax = nn.Softmax(dim=1)(mixed_outputs)
        re_loss = (- mixed_t * torch.log(softmax + epsilon)).sum(dim=1)
        re_loss = re_loss.mean(dim=0)
        return loss, re_loss, W_tt

