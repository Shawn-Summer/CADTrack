# CADTrack 中 3 个模块的解读

```python
for i, blk in enumerate(self.blocks):
    x_r, x_x = blk(x_r, x_x)

    if self.cross_loc is not None and i in self.cross_loc:
        x_r, x_x = self.MCI[cross_index](x_r, x_x)
        cross_index += 1

    if i < self.depth-1:
        proj_x_r = self.moe_proj_layers[i](x_r)
        proj_x_x = self.moe_proj_layers[i](x_x)
    else:
        proj_x_r = x_r
        proj_x_x = x_x
    res_x_r.append(x_r[:, lens_z+1:, :])
    res_x_x.append(x_x[:, lens_z+1:, :])
    res_proj_x_r.append(proj_x_r[:, lens_z+1:, :])
    res_proj_x_x.append(proj_x_x[:, lens_z+1:, :])
```

以上代码中，x_r 和 x_x 分别是 rgb 和 x 模态的 tokens，形状都是 [B,1+L_z+L_x,C],其中 1 是 vit 的cls_token,论文中直接用了时空线索，L_z 是 64, L_x 是 256, C 是 768 
在 cross_index = [4,7,10] 即在 这几层后连接一个 MCI 的module，即paper中的 MFI 模块。

res_proj_x_r 是 rgb 模态的经过 shared 线性映射后的 不同层的feature, res_x_r 则是原始特征向量。并且他们只是 search region 的部分。 这个东西是用来做 CAM 模块的。

## MFI 模块

位于 backbone 中 blk 的 4,7,10 之后，用于跨模态交互，

```python
class MCILayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adap_decline_x = nn.Linear(dim, 8)
        self.adap_decline_xi = nn.Linear(dim, 8)
        self.adap_share = Mamba_adapter(dim=8)
        self.adap_incline_x = nn.Linear(8, dim)
        self.adap_incline_xi = nn.Linear(8, dim)

    def forward(self, x, xi):
        x_adap = self.adap_decline_x(x)
        xi_adap = self.adap_decline_xi(xi)
        cat_adap = self.adap_share(x_adap, xi_adap) # Mamba核心交互
        x_adap, xi_adap = torch.split(cat_adap, cat_adap.shape[1] // 2, dim=1)
        x_adap = self.adap_incline_x(x_adap)
        xi_adap = self.adap_incline_xi(xi_adap)
        x = x + x_adap
        xi = xi + xi_adap

        return x, xi
```

这里先对 x 和 xi 两个模态tokens做降维，从768降维到8,然后进入mamba模块交互，然后 分割，在升维，然后做一个残差连接。
其中mamba交互核心的逻辑是：

```python
class Mamba_adapter(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.MA = nn.ModuleList([
            MABlock(
                hidden_dim=dim,
                mlp_ratio=0.0,
                d_state=16, )
            for i in range(2)])

    def forward(self, x_adap, xi_adap):

        x_down = torch.cat([x_adap, xi_adap], 1)

        for i in range(2):
            x_down_flip = x_down.flip(dims=[1])
            x_down= self.MA[i](x_down)
            x_down_flip = self.MA[i](x_down_flip)
            x_down_flip = x_down_flip.flip(dims=[1])
            x_down = x_down + x_down_flip

        return x_down
```
这里是一个双向mamba的逻辑

## CAM 模块

```python
        shift_indice_x_r, selected_proj_x_r = self.shift(res_x_r, res_proj_x_r)
        selected_proj_x_r = selected_proj_x_r.sum(dim=1)
        x_r = torch.cat([x_r[:, :lens_z+1, :], selected_proj_x_r], dim=1)

        shift_indice_x_x, selected_proj_x_x = self.shift(res_x_x, res_proj_x_x)
        selected_proj_x_x = selected_proj_x_x.sum(dim=1)
        x_x = torch.cat([x_x[:, :lens_z+1, :], selected_proj_x_x], dim=1)
```

以上，把各layer原始特征作为guidance，用于稀疏选择映射后的特征，他们的shape 都是 [B,256,768] 就是其中的 self.shift(res_x_r, res_proj_x_r),具体如下：

```python
def shift(self, guide, x):
        x_tensor = torch.stack(x, dim=1)
        B, _, L, C = x_tensor.shape

        guide = torch.cat(guide[1:-1], 1)
        guide = guide.permute(0, 2, 1)
        x_avg = self.moe_avg(guide)  # B, C, 1280
        logit = self.moe_mlp(x_avg)  # B, C, 128
        logit = self.moe_fc(logit.reshape(B, -1)) # B，10
        _, shift_indice = torch.sort(logit, dim=1)
        shift_indice = shift_indice + 1

        zeros = torch.zeros((B, 1), dtype=shift_indice.dtype, device=shift_indice.device)
        elevens = torch.full((B, 1), 11, dtype=shift_indice.dtype, device=shift_indice.device)
        top_4 = shift_indice[:, :4]
        combined = torch.cat([zeros, top_4, elevens], dim=1)
        sorted_combined, _ = torch.sort(combined, dim=1)
        sorted_combined_expanded = sorted_combined.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, C)
        selected_layers = torch.gather(x_tensor, 1, sorted_combined_expanded)
        proj_weights = self.moe_proj_weights.expand(B, -1)
        selected_proj = selected_layers * proj_weights.unsqueeze(-1).unsqueeze(-1)

        return sorted_combined, selected_proj
```

具体做法是，选择guide中间10个层的特征，做计算得到 logit （B，10）,然后找到其 top_4 的index，然后和第一层和最后一层连接起来，即 selected_layers，对于被选中的6个layer，再使用一个可学习的weight来给他乘起来。 最终输出的是 selected_proj ,shape [B,6,256,768]

对于 rgb 模态的 cross layer feature，就是直接 6个layer sum 一下用来聚合不同层次的特征，然后和原始的 template 和 cue 的token 连接起来，即得到 x_r ，同理红外模态。

## DAM 模块

```python
for i in range(len(search)):
            x, aux_dict, len_zx = self.backbone(z=template, x=search[i], track_query_before=track_query_before
                                                )
            num_template_token = len_zx[0]
            num_search_token = len_zx[1]

            B, N, _ = x.size()
            temp_r = x[:, :N // 2, :]
            temp_x = x[:, N // 2:, :]

            boss_fea = torch.cat([temp_r[:, :1, :], temp_x[:, :1, :]], dim=1) # cues [B,2,C]
            temp_r_str, temp_x_str = self.CDA(temp_r[:, 1:num_template_token // 2 + 1, :],
                                              temp_r[:, num_template_token // 2 + 1:num_template_token + 1, :],
                                              temp_x[:, 1:num_template_token // 2 + 1, :],
                                              temp_x[:, num_template_token // 2 + 1:num_template_token + 1, :],
                                              boss_fea) # temp_r_str : cues of r [B,1,C]
            temp_r_query = temp_r_str.clone().detach()
            temp_x_query = temp_x_str.clone().detach()
            track_query_before = [temp_r_query, temp_x_query]
            
            # use cues strengthen the output feature 
            feat_last_r = temp_r[:, -num_search_token:, :]
            temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_last_r, feat_last_r)
            temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
            att_r = torch.matmul(feat_last_r, temp_attn_r.transpose(1, 2))
            feat_last_r = att_r * feat_last_r

            feat_last_x = temp_x[:, -num_search_token:, :]
            temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_last_x, feat_last_x)
            temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
            att_x = torch.matmul(feat_last_x, temp_attn_x.transpose(1, 2))
            feat_last_x = att_x * feat_last_x

            feat_last = torch.cat([feat_last_r, feat_last_x], dim=-1) # [B,256,768*2]

            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out['track_query_before'] = track_query_before
            out['backbone_feat'] = feat_last
            out_dict.append(out)
```

上面代码中的，CDA 模块，就是空间可对齐模块，输入是2个模态的 initial template 和 dynamic template以及 cues，输出是增强后的 cues。然后后续使用 cues 继续增强 output feature

这里的 CDA 逻辑如下：

```python
def forward(self, x_r1, x_r2, x_x1, x_x2, boss, writer=None, epoch=None, img_path=None, texts=''):
        x_r1 = x_r1.reshape(x_r1.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_r2 = x_r2.reshape(x_r2.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_x1 = x_x1.reshape(x_x1.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_x2 = x_x2.reshape(x_x2.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_r1_blocks = self.split_into_blocks_with_overlap(x_r1, self.window_size, self.stride_block).flatten(1, 2)
        x_r2_blocks = self.split_into_blocks_with_overlap(x_r2, self.window_size, self.stride_block).flatten(1, 2)
        x_x1_blocks = self.split_into_blocks_with_overlap(x_x1, self.window_size, self.stride_block).flatten(1, 2)
        x_x2_blocks = self.split_into_blocks_with_overlap(x_x2, self.window_size, self.stride_block).flatten(1, 2)
        boss = boss.permute(0, 2, 1).unsqueeze(-2)
        query_cash = []
        for i in range(self.num_da):
            query_cash.append(
                self.da_group[i](boss, x_r1_blocks[:, i], x_r2_blocks[:, i], x_x1_blocks[:, i], x_x2_blocks[:, i], writer=writer, epoch=epoch,
                                 img_path=img_path, text=texts).squeeze(-1))
        fea = query_cash[0].permute(0, 2, 1)
        track_x, track_r = torch.split(fea, fea.size(1) // 2, dim=1)

        return track_x, track_r
```

先分块，然后分块学习offset，实际上做空间矫正的逻辑本质是把4个template映射到同一个空间中



