# 15分钟了解Pytorch Lightning

**前置知识**：推荐pytorch
**目标**：通过PL中7个关键步骤了解PL工作流程

PL是基于pytorch的高层API，自带丰富的工具为AI学者和工程师快速创建高性能模型，去除繁琐的重复流程同时保持灵活性。

<iframe src="https://pl-public-data.s3.amazonaws.com/assets_lightning/pl_readme_gif_2_0.m4v" scrolling="no" width="800px" height="600px" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

使用组织好的pytorch代码，PL可以：

1. 避免重复流程。比如gpu设置，device设置，backward()等。
2. 高可读性并更容易复现。PL将代码按照运行周期设置了不同HOOK，你可以很容易地找到关键代码。然后将其它方法迁移复现。比如损失计算在training_step()方法中。
3. 简单设置即可使用多GPU策略。只需对Trainer配置设置和策略，无需在模型部分操作。
4. 快速的test流程。无需额外写测试逻辑，只需和train流程类似，实现test_step()，内部完成推理和保存。调用trainer.test()即可完成测试流程。

## 第一步 安装PL

pip 安装

```shell
pip install lightning
``` 

conda 安装
```shell
conda install lightning -c conda-forge
```

## 第二步 定义一个LightningModule子类

子类中可以使用原生pytorch nn.Module创建的模块搭建PL的模型，然后在training_step()方法中实现损失计算的过程。将[LightningModule](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html)的子类传入[Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html)后，即会自动调用计算损失并反向传播。
```python
# 导入包
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# 定义一个线性层和激活函数组成的编码器-解码器，你可以使用任何nn.Module创建的模块
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# 定义一个继承LightningModule的子类
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        # 设置属性，模型包含两个子模块
        self.encoder = encoder
        self.decoder = decoder

    # 单步训练过程，里面包含forward流程，可以单独写出
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # 计算损失
        loss = nn.functional.mse_loss(x_hat, x)
        # 记录训练过程中的损失。如果要使用tensorboard,wandb等工具，需要在trainer创建时指定，并如下设置logger=Ture
        self.log("train_loss", loss, logger=Ture)
        return loss
    # 配置优化器
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 创建一个模型
autoencoder = LitAutoEncoder(encoder, decoder)
```

## 第三步 定义一个数据集加载器

任何可迭代iterable 的对象（list, dataloader, dict, numpy等）均可作为加载器。

```python
# 这里使用的torchvision中包含的MNIST数据集，可以自定义，方法和pytorch一样实现get_item()方法。
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
```

## 第四步 训练模型

创建一个训练管理器trainer，调用fit()方法时传入模型和数据集加载器，开始训练。

```python
# 创建Trainer时有不同参数，详见API说明
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
# 传入模型和数据集加载器并开始训练
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

Trainer 包含train/validation/test loop，日志记录、gpu设置、模型保存策略
等逻辑，避免花费过多精力在重复的流程上。

## 第五步 使用模型

训练完成后可以将模型转换格式部署至生产环境中，也只可加载权重预测。

```
# 加载checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# 预测
fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
```

## 第六步 训练监控

如果创建Trainer时指定了logger，如tensorboard, wandb并给定相关设置，即可打开日志文件查看训练状态。比如使用了tensorboard作为logger，执行以下命令后，浏览器打开 http://localhost:6006/

```shell
cd path/to/your/log/file
tensorboard --logdir .
```

## 第七步 训练的高级功能

在创建Trainer时你可以使用不同的设备（cpu、gpu、tpu等）、不同的策略、精度，以及callback。

```python
# 4块GPU上训练
trainer = Trainer(
    devices=4,
    accelerator="gpu",
 )

# 通过Deepspeed/fsdp策略，在16精度下由4块GPU训练
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision=16
 )

# Trainer 中20多个有用的配置项，比如
trainer = Trainer(
    max_epochs=10,# 最大训练10个epochs
    min_epochs=5,# 最小训练5个epoch
    overfit_batches=0.01 # 使用0.01的数据训练，以快速测试代码，默认为1
 )

# 将实现了callback接口的模块整合进流程中，callback中包含不同流程的处理逻辑
trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])
```

参考文档
1. [Pytorch Lightning](https://lightning.ai/docs/pytorch/latest/starter/introduction.html)
