# 代码中架构问题

* 问题1: 数据集的结构是怎么样的，一个样本的数据有哪些？
  数据集中包括很多序列，即视频，每个序列中有红外图片，rgb图片以及对应的检测框标注。

  

* 问题2: 搜索帧和模板帧怎么来的？
搜索帧和模板帧都是从图片中裁剪出来的，根据中心和size裁剪的；搜索帧和模板帧都是选自于同一个序列，一般来说，模板桢的id会在搜索桢之前。
训练阶段，模板桢不区分动态和初始模板，随机采样两个模板；
测试阶段，动态模板是根据检测结果生成的，逻辑如下：有个更新间隔和阈值啥的，并且在测试文件的初始化的时候，把初始模板复制后作为动态模板

```python
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):

                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)
```

* 问题3：多帧训练和测试阶段的online track相关？
  动态模板：
    训练阶段，采用时序监督训练（多帧训练，帧数为3），模板数量是2,不区分动态和初始模板
    测试阶段，采用逐帧推理，遍历序列中所有帧，逐一作为搜索桢，模板数量是2（两个相同的模板），其中动态模板会逐帧更新，
  时序cues：
    训练阶段，由于是多帧训练，时序cues利用了
    测试阶段，在逐帧推理后，会在`out_dict`中找到对应的cues，传递给下一帧

  

* 问题4: 模型的输入输出的参数相关
experiments/cadtrack/cadtrack.yaml 是实验采用的配置文件，而 lib/config/cadtrack/config.py 是默认配置。
template size 是 128x128；search size 是 256x256；stride 是 16；
template token number 是 64；search token number 是 256
而embed_dim与选取的backbone有关，base 是768, small是384, tiny是192, 具体选择在 配置文件中的 `MODEL.BACKBONE.TYPE`



