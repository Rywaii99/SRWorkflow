import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch_util import default_conv, Upsample, ResBlock, BasicBlock
from utils.registry import ARCH_REGISTRY


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=default_conv,
                 res_scale=1):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        # number of hash buckets/hash bits
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret


@ARCH_REGISTRY.register()
class NLSN(nn.Module):
    """
    NLSN 类：非局部稀疏注意力网络（Non-Local Sparse Attention Network）。
    通过非局部稀疏注意力模块和残差块实现图像的超分辨率重建。
    """

    def __init__(self, n_resblocks=32, n_feats=256, scale=2, res_scale=0.1, n_hashes=4, chunk_size=144, n_colors=3, conv=default_conv):
        super(NLSN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # 定义头部模块
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # 定义主体模块
        m_body = [NonLocalSparseAttention(
            channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale)]

        for i in range(n_resblocks):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i + 1) % 8 == 0:
                m_body.append(NonLocalSparseAttention(
                    channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # 定义尾部模块
        m_tail = [
            Upsample(scale, n_feats),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))