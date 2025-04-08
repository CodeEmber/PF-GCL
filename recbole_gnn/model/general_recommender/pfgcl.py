import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_gnn.model.general_recommender import LightGCN


class L2RegLoss(nn.Module):

    def __init__(self):
        super(L2RegLoss, self).__init__()

    def forward(self, *args) -> torch.Tensor:
        """计算L2正则化损失

        Args:
            *args: 需要计算正则化的embedding

        Returns:
            torch.Tensor: L2正则化损失
        """
        batch_size = args[0].shape[0]
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2) ** 2
        emb_loss = emb_loss / 2
        return emb_loss / batch_size


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """计算BPR损失

        Args:
            user_emb (torch.Tensor): 用户embedding
            pos_emb (torch.Tensor): 正样本embedding
            neg_emb (torch.Tensor): 负样本embedding

        Returns:
            torch.Tensor: 损失
        """
        pos_score = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_item_emb, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))
        return loss


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        fn: str = "mean",
    ) -> torch.Tensor:
        """计算InfoNCE损失

        Args:
            view1 (torch.Tensor): 第一个视图的嵌入
            view2 (torch.Tensor): 第二个视图的嵌入

        Returns:
            torch.Tensor: InfoNCE损失
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        pos_score = pos_score.repeat(2)

        ttl_score1 = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score1 = torch.exp(ttl_score1 / self.temperature).sum(dim=1)
        ttl_score2 = torch.matmul(view2, view1.transpose(0, 1))
        ttl_score2 = torch.exp(ttl_score2 / self.temperature).sum(dim=1)
        ttl_score = torch.cat([ttl_score1, ttl_score2])
        cl_loss = -torch.log(pos_score / ttl_score)
        if fn == "mean":
            return torch.mean(cl_loss)
        elif fn == "sum":
            return torch.sum(cl_loss)
        else:
            raise ValueError("fn must be 'mean' or 'sum'")


class PFGCL(LightGCN):

    def __init__(self, config, dataset):
        super(PFGCL, self).__init__(config, dataset)
        self.temperature = config["temperature"]
        self.lmbd_reg = config["lmbd_reg"]
        self.lmbd_ssl = config["lmbd_ssl"]
        self.bpr_loss = BPRLoss()
        self.reg_loss = L2RegLoss()
        self.infonce_loss = InfoNCELoss(self.temperature)
        self.multi_modal_attention = MultiModalAttention(config["embedding_size"])

    def forward(self, is_train=False):
        all_embs = self.get_ego_embeddings()
        embs_list = [all_embs]
        for i in range(2):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            embs_list.append(all_embs)
        embedding_dict = dict()
        for i in range(len(embs_list)):
            embedding_dict["embedding_{}".format(i)] = embs_list[i]
            embedding_dict["embedding_{}_user".format(i)] = embs_list[i][: self.n_users]
            embedding_dict["embedding_{}_item".format(i)] = embs_list[i][self.n_users :]
        embedding_dict["embedding_all"] = torch.mean(
            torch.stack(embs_list, dim=1), dim=1
        )
        embedding_dict["embedding_all_user"] = embedding_dict["embedding_all"][
            : self.n_users
        ]
        embedding_dict["embedding_all_item"] = embedding_dict["embedding_all"][
            self.n_users :
        ]
        if is_train:
            return embedding_dict
        else:
            embedding_all = torch.mean(torch.stack(embs_list, dim=1), dim=1)
            user_all_embeddings = embedding_all[: self.n_users]
            item_all_embeddings = embedding_all[self.n_users :]
            return user_all_embeddings, item_all_embeddings

    def create_infonce_loss(self, data, embedding_dict):
        u_idx = data[self.USER_ID]
        i_idx = data[self.ITEM_ID]
        # user_embedding_list = [embedding_dict["embedding_0_user"][u_idx], embedding_dict["embedding_1_item"][i_idx], embedding_dict["embedding_2_user"][u_idx]]
        # item_embedding_list = [embedding_dict["embedding_0_item"][i_idx], embedding_dict["embedding_1_user"][u_idx], embedding_dict["embedding_2_item"][i_idx]]
        # user_embedding = torch.mean(torch.stack(user_embedding_list, dim=1), dim=1)
        # pos_item_embedding = torch.mean(torch.stack(item_embedding_list, dim=1), dim=1)
        user_cl_loss = self.infonce_loss(
            embedding_dict["embedding_0_user"][u_idx],
            embedding_dict["embedding_1_item"][i_idx],
        )
        item_cl_loss = self.infonce_loss(
            embedding_dict["embedding_0_item"][i_idx],
            embedding_dict["embedding_1_user"][u_idx],
        )

        loss1 = 0.5 * user_cl_loss + 0.5 * item_cl_loss

        user_cl_loss = self.infonce_loss(
            embedding_dict["embedding_2_user"][u_idx],
            embedding_dict["embedding_1_item"][i_idx],
        )
        item_cl_loss = self.infonce_loss(
            embedding_dict["embedding_2_item"][i_idx],
            embedding_dict["embedding_1_user"][u_idx],
        )

        loss2 = 0.5 * user_cl_loss + 0.5 * item_cl_loss

        user_cl_loss = self.infonce_loss(
            embedding_dict["embedding_0_user"][u_idx],
            embedding_dict["embedding_2_user"][u_idx],
        )
        item_cl_loss = self.infonce_loss(
            embedding_dict["embedding_0_item"][i_idx],
            embedding_dict["embedding_2_item"][i_idx],
        )

        loss3 = 0.5 * user_cl_loss + 0.5 * item_cl_loss
        return loss1, loss2, loss3

    def total_loss(self, data, embedding_dict):
        u_idx = data[self.USER_ID]
        i_idx = data[self.ITEM_ID]
        n_i_idx = data[self.NEG_ITEM_ID]
        # user_embedding_list = [embedding_dict["embedding_0_user"][u_idx], embedding_dict["embedding_1_item"][i_idx], embedding_dict["embedding_2_user"][u_idx]]
        # item_embedding_list = [embedding_dict["embedding_0_item"][i_idx], embedding_dict["embedding_1_user"][u_idx], embedding_dict["embedding_2_item"][i_idx]]
        # neg_item_embedding_list = [embedding_dict["embedding_0_item"][n_i_idx], embedding_dict["embedding_1_user"][u_idx], embedding_dict["embedding_2_item"][n_i_idx]]
        # user_embedding = torch.mean(torch.stack(user_embedding_list, dim=1), dim=1)
        # pos_item_embedding = torch.mean(torch.stack(item_embedding_list, dim=1), dim=1)
        # neg_item_embedding = torch.mean(torch.stack(neg_item_embedding_list, dim=1), dim=1)

        user_embedding = embedding_dict["embedding_all_user"][data[self.USER_ID]]
        pos_item_embedding = embedding_dict["embedding_all_item"][data[self.ITEM_ID]]
        neg_item_embedding = embedding_dict["embedding_all_item"][
            data[self.NEG_ITEM_ID]
        ]

        user_ego_embedding = self.user_embedding(u_idx)
        pos_item_ego_embedding = self.item_embedding(i_idx)
        neg_item_ego_embedding = self.item_embedding(n_i_idx)

        bpr_loss = self.bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        reg_loss = self.reg_loss(
            user_ego_embedding, pos_item_ego_embedding, neg_item_ego_embedding
        )
        cl_loss1, cl_loss2, cl_loss3 = self.create_infonce_loss(data, embedding_dict)
        return bpr_loss, reg_loss, cl_loss1, cl_loss2, cl_loss3

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        embedding_dict = self.forward(True)
        bpr_loss, reg_loss, cl_loss1, cl_loss2, cl_loss3 = self.total_loss(
            interaction, embedding_dict
        )
        # output_dict = {
        #     "bpr_loss": bpr_loss,
        #     "reg_loss": reg_loss,
        #     "cl_loss1": cl_loss1,
        #     "cl_loss2": cl_loss2,
        #     "cl_loss3": cl_loss3,
        #     "lmbd_ssl": self.lmbd_ssl,
        #     "lmbd_reg": self.lmbd_reg,
        # }
        # output_dict["embedding_dict"] = embedding_dict
        loss = (
            bpr_loss
            + self.lmbd_reg * reg_loss
            + self.lmbd_ssl * torch.mean(cl_loss1 + cl_loss2 + cl_loss3)
        )
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(False)
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)


class MetaWeight(nn.Module):

    def __init__(self, hidden_dim: int = 64):
        super(MetaWeight, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(0.5)
        self.bn1_user = nn.BatchNorm1d(hidden_dim)
        self.bn1_item = nn.BatchNorm1d(hidden_dim)
        self.layer1_user = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layer2_user = nn.Linear(hidden_dim, 3)

        self.layer1_item = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layer2_item = nn.Linear(hidden_dim, 3)

        nn.init.xavier_uniform_(self.layer1_user.weight)
        nn.init.xavier_uniform_(self.layer2_user.weight)
        nn.init.xavier_uniform_(self.layer1_item.weight)
        nn.init.xavier_uniform_(self.layer2_item.weight)

    def forward(self, output, u_idx, i_idx) -> torch.Tensor:
        embedding_0_user = output["embedding_dict"]["embedding_0_user"][u_idx]
        embedding_0_item = output["embedding_dict"]["embedding_0_item"][i_idx]
        embedding_1_user = output["embedding_dict"]["embedding_1_item"][i_idx]
        embedding_1_item = output["embedding_dict"]["embedding_1_user"][u_idx]
        embedding_2_user = output["embedding_dict"]["embedding_2_user"][u_idx]
        embedding_2_item = output["embedding_dict"]["embedding_2_item"][i_idx]

        user_embedding = torch.cat(
            [embedding_0_user, embedding_1_user, embedding_2_user], dim=1
        )
        item_embedding = torch.cat(
            [embedding_0_item, embedding_1_item, embedding_2_item], dim=1
        )

        user_embedding = self.layer1_user(user_embedding)
        user_embedding = self.bn1_user(user_embedding)
        user_embedding = self.dropout(user_embedding)
        user_embedding = self.layer2_user(user_embedding)
        user_embedding = self.sigmoid(user_embedding)
        user_embedding = torch.mean(user_embedding, dim=0)
        user_embedding = self.softmax(user_embedding)

        item_embedding = self.layer1_item(item_embedding)
        item_embedding = self.bn1_item(item_embedding)
        item_embedding = self.dropout(item_embedding)
        item_embedding = self.layer2_item(item_embedding)
        item_embedding = self.sigmoid(item_embedding)
        item_embedding = torch.mean(item_embedding, dim=0)
        item_embedding = self.softmax(item_embedding)

        return torch.mean(torch.stack([user_embedding, item_embedding]), dim=0)


class MultiModalAttention(nn.Module):

    def __init__(self, embedding_dim):
        super(MultiModalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        # 定义一个全连接层来计算注意力权重
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self, i_emb_u, v_emb_u, t_emb_u):

        # 计算每个模态的注意力权重
        att_i = self.attention_fc(i_emb_u).squeeze(-1)
        att_v = self.attention_fc(v_emb_u).squeeze(-1)
        att_t = self.attention_fc(t_emb_u).squeeze(-1)

        # 使用softmax函数归一化注意力权重
        attention_weights = F.softmax(torch.stack([att_i, att_v, att_t]), dim=0)

        # 将注意力权重切分成原来的三部分
        att_weights_i = attention_weights[0]
        att_weights_v = attention_weights[1]
        att_weights_t = attention_weights[2]

        # 进行加权融合
        weighted_i_emb_u = att_weights_i.unsqueeze(-1) * i_emb_u
        weighted_v_emb_u = att_weights_v.unsqueeze(-1) * v_emb_u
        weighted_t_emb_u = att_weights_t.unsqueeze(-1) * t_emb_u

        # 融合所有模态的信息
        fused_user_embedding = weighted_i_emb_u + weighted_t_emb_u + weighted_v_emb_u
        # 保存权重用于后续绘图
        self.weights = attention_weights.detach().cpu().numpy()

        return fused_user_embedding
