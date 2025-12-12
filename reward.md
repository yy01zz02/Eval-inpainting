**3\. 基础架构模型：CLIP-ViT-Large-Patch14**

### **3.1 模型概览与理论基础**

**OpenAI CLIP (Contrastive Language-Image Pre-training)** 是当前绝大多数文生图评估模型的基石。用户提供的 openai/clip-vit-large-patch14 是CLIP家族中的大参数量版本（ViT-L/14），其视觉编码器将图像分割为14x14像素的图块（Patch），并通过Transformer层提取特征。

该模型本身即可通过计算图像嵌入（Image Embedding）与文本嵌入（Text Embedding）的余弦相似度（Cosine Similarity）来作为图文一致性（CLIP Score）的评分标准。同时，它也是后续要讨论的 **Aesthetic Predictor** 和 **PickScore** 的骨干网络。

### **3.2 本地路径分析**

用户提供的路径为：  
/home/admin/workspace/aop\_lab/app\_data/.cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41  
这是一个标准的 Hugging Face Hub 缓存结构中的 **Snapshot（快照）** 目录。在HF的缓存机制中，snapshots 目录下存放的是具体版本的模型文件（通常是软链接指向 blobs 目录，但在拷贝环境下可能是实体文件）。该目录下通常包含 config.json, pytorch\_model.bin (或 model.safetensors), tokenizer.json, preprocessor\_config.json 等关键文件。

### **3.3 本地调用实现代码**

为了在本地调用该模型，我们需要使用 transformers 库，并显式指定 local\_files\_only=True 或者直接传入绝对路径。

Python

import torch  
from PIL import Image  
from transformers import CLIPProcessor, CLIPModel

class LocalCLIPLoader:  
    def \_\_init\_\_(self, model\_path, device="cuda"):  
        self.model\_path \= model\_path  
        self.device \= device  
        print(f"\[CLIP-L\] 正在从本地路径加载: {self.model\_path}")  
          
        try:  
            \# 加载模型与处理器  
            self.model \= CLIPModel.from\_pretrained(  
                self.model\_path,   
                local\_files\_only=True  
            ).to(self.device).eval()  
              
            self.processor \= CLIPProcessor.from\_pretrained(  
                self.model\_path,   
                local\_files\_only=True  
            )  
            print("\[CLIP-L\] 加载成功。")  
        except OSError as e:  
            print(f"\[CLIP-L\] 加载失败，请检查路径是否包含 config.json 与权重文件。错误信息: {e}")  
            raise

    def score(self, image\_input, text\_input):  
        """  
        计算基础的 CLIP Score (余弦相似度)  
        """  
        if isinstance(image\_input, str):  
            image \= Image.open(image\_input).convert("RGB")  
        else:  
            image \= image\_input

        \# 预处理  
        inputs \= self.processor(  
            text=\[text\_input\],   
            images=image,   
            return\_tensors="pt",   
            padding=True  
        ).to(self.device)

        with torch.no\_grad():  
            outputs \= self.model(\*\*inputs)  
            \# 归一化特征向量  
            image\_embeds \= outputs.image\_embeds / outputs.image\_embeds.norm(dim=-1, keepdim=True)  
            text\_embeds \= outputs.text\_embeds / outputs.text\_embeds.norm(dim=-1, keepdim=True)  
              
            \# 计算余弦相似度  
            score \= (text\_embeds @ image\_embeds.T).item()  
              
        return score, image\_embeds  \# 返回 embedding 供后续模型使用

\# 实例化示例  
if \_\_name\_\_ \== "\_\_main\_\_":  
    clip\_path \= "/home/admin/workspace/aop\_lab/app\_data/.cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"  
    clip\_loader \= LocalCLIPLoader(clip\_path)  
    \# score \= clip\_loader.score("test.jpg", "a test prompt")

## ---

**4\. 线性美学评分：LAION-AI Aesthetic Predictor**

### **4.1 模型概览与理论基础**

**LAION-Aesthetics Predictor** 是一个极简但高效的模型。它不是一个完整的深度神经网络，而是一个**线性探针（Linear Probe）**——即一个简单的多层感知机（MLP），通常只有一层线性层。它建立在 CLIP ViT-L/14 的图像嵌入之上。

其训练数据来自 LAION-5B 数据集的一个子集，标签是人类对图像美学质量的评分（1-10分）。该模型不考虑文本提示，仅评估图像本身的构图、色彩、光影等美学属性。

### **4.2 文件与依赖分析**

用户拥有的文件是 sa\_0\_4\_vit\_l\_14\_linear.pth。

* 这是一个 PyTorch 的权重字典文件（State Dict）。  
* 它依赖于上文提到的 **CLIP-ViT-Large-Patch14** 来提取 768 维的特征向量。  
* 该权重对应的模型结构是一个输入维度为 768，输出维度为 1 的线性层。

### **4.3 本地调用实现代码**

由于没有现成的 pip 包来直接加载这个 .pth 文件，我们需要手动定义模型结构并加载权重。

Python

import torch  
import torch.nn as nn

class AestheticPredictor(nn.Module):  
    def \_\_init\_\_(self, input\_size):  
        super().\_\_init\_\_()  
        self.layers \= nn.Sequential(  
            nn.Linear(input\_size, 1)  
        )

    def forward(self, x):  
        return self.layers(x)

class LocalAestheticScorer:  
    def \_\_init\_\_(self, clip\_loader, weight\_path, device="cuda"):  
        """  
        clip\_loader: 上一节实例化的 LocalCLIPLoader 对象  
        weight\_path: sa\_0\_4\_vit\_l\_14\_linear.pth 的本地路径  
        """  
        self.device \= device  
        self.clip\_loader \= clip\_loader  
          
        \# 定义线性层结构，CLIP ViT-L/14 的输出维度是 768  
        self.model \= AestheticPredictor(768).to(device).eval()  
          
        print(f"\[Aesthetic\] 正在加载线性层权重: {weight\_path}")  
        try:  
            state\_dict \= torch.load(weight\_path, map\_location=device)  
            \# 有些版本的保存方式可能是直接保存了 linear.weight，需要适配  
            \# 这里的适配逻辑基于常见结构，如果 key 不匹配可能需要调整  
            new\_state\_dict \= {}  
            for k, v in state\_dict.items():  
                if "weight" in k: new\_state\_dict\["layers.0.weight"\] \= v  
                if "bias" in k: new\_state\_dict\["layers.0.bias"\] \= v  
              
            \# 如果上面逻辑太复杂，可以直接尝试 load，失败再调整  
            try:  
                self.model.load\_state\_dict(state\_dict)  
            except:  
                \# 针对 sa\_0\_4\_vit\_l\_14\_linear.pth 的特定结构（通常就是一个 state dict）  
                \# 它的 key 通常是 "weight" 和 "bias"  
                self.model.layers.weight.data \= state\_dict\['weight'\]  
                self.model.layers.bias.data \= state\_dict\['bias'\]  
                  
            print("\[Aesthetic\] 权重加载成功。")  
        except Exception as e:  
            print(f"\[Aesthetic\] 权重加载失败: {e}")  
            raise

    def score(self, image\_input):  
        \# 1\. 使用 CLIP 提取特征  
        \# 注意：Aesthetic Predictor 只需要图像特征，不需要文本  
        \# 我们复用 clip\_loader 的处理逻辑，传入空文本或忽略文本输出  
        if isinstance(image\_input, str):  
            image \= Image.open(image\_input).convert("RGB")  
        else:  
            image \= image\_input  
              
        inputs \= self.clip\_loader.processor(images=image, return\_tensors="pt").to(self.device)  
          
        with torch.no\_grad():  
            vision\_outputs \= self.clip\_loader.model.get\_image\_features(\*\*inputs)  
            \# 关键步骤：归一化  
            vision\_embeds \= vision\_outputs / vision\_outputs.norm(dim=-1, keepdim=True)  
              
            \# 2\. 线性层打分  
            prediction \= self.model(vision\_embeds)  
              
        return prediction.item()

\# 实例化示例  
\# aesthetic\_path \= "/path/to/sa\_0\_4\_vit\_l\_14\_linear.pth"  
\# aesthetic\_scorer \= LocalAestheticScorer(clip\_loader, aesthetic\_path)

## ---

**5\. 人类偏好微调：yuvalkirstain/PickScore\_v1**

### **5.1 模型概览与理论基础**

**PickScore** 是首批在大型人类偏好数据集（Pick-a-Pic dataset）上进行微调的 CLIP 模型之一。它的核心思想是通过微调 CLIP 的权重，使其输出的图文相似度分数能够尽可能符合人类的选择偏好。与原始 CLIP 相比，PickScore 更能反映图像的质量和对提示词的遵循程度，而不仅仅是语义匹配。

### **5.2 本地路径分析**

路径：/home/admin/workspace/aop\_lab/app\_data/.cache/models--yuvalkirstain--PickScore\_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800

这也是一个标准的 Hugging Face Snapshot。PickScore 基于 **CLIP-ViT-Huge-14** 架构（注意：是 Huge 不是 Large，因此不能复用前面的 CLIP-Large 模型）。该目录下应包含适用于 CLIP-Huge 的权重文件。

### **5.3 本地调用实现代码**

PickScore 的调用方式与标准 CLIP 类似，但必须使用其专有的处理器（Processor）配置，因为 CLIP-Huge 的预处理参数（如分辨率）可能不同。

Python

from transformers import AutoProcessor, AutoModel

class LocalPickScoreLoader:  
    def \_\_init\_\_(self, model\_path, device="cuda"):  
        self.device \= device  
        self.model\_path \= model\_path  
        print(f" 正在加载: {model\_path}")  
          
        try:  
            \# 自动加载配置和模型  
            self.processor \= AutoProcessor.from\_pretrained(model\_path, local\_files\_only=True)  
            self.model \= AutoModel.from\_pretrained(model\_path, local\_files\_only=True).eval().to(device)  
            print(" 加载成功。")  
        except Exception as e:  
            print(f" 加载失败: {e}")  
            raise

    def score(self, image\_input, prompt):  
        if isinstance(image\_input, str):  
            image \= Image.open(image\_input).convert("RGB")  
        else:  
            image \= image\_input

        \# 预处理  
        inputs \= self.processor(  
            images=image,  
            text=prompt,  
            padding=True,  
            truncation=True,  
            max\_length=77,  
            return\_tensors="pt",  
        ).to(self.device)

        with torch.no\_grad():  
            \# 获取特征  
            image\_embs \= self.model.get\_image\_features(\*\*inputs)  
            image\_embs \= image\_embs / torch.norm(image\_embs, dim=-1, keepdim=True)  
          
            text\_embs \= self.model.get\_text\_features(\*\*inputs)  
            text\_embs \= text\_embs / torch.norm(text\_embs, dim=-1, keepdim=True)  
              
            \# PickScore 的计算方式：logit\_scale \* (text @ image.T)  
            \# 这是一个标量分数，分数越高代表偏好度越高  
            scores \= self.model.logit\_scale.exp() \* (text\_embs @ image\_embs.T)  
              
        return scores.item()

\# 实例化示例  
if \_\_name\_\_ \== "\_\_main\_\_":  
    pick\_path \= "/home/admin/workspace/aop\_lab/app\_data/.cache/models--yuvalkirstain--PickScore\_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800"  
    pick\_scorer \= LocalPickScoreLoader(pick\_path)

## ---

**6\. 进阶偏好评分：xswu/HPSv2**

### **6.1 模型概览与理论基础**

**HPSv2 (Human Preference Score v2)** 是目前学术界广泛使用的基准模型之一。它基于 HPDv2 数据集训练，该数据集包含约 80 万个人类偏好选择。HPSv2 同样基于 CLIP 架构（OpenCLIP ViT-H/14），旨在解决 HPSv1 对真实图片和动漫图片泛化能力不足的问题。

### **6.2 本地路径与“库”的博弈**

路径：/home/admin/workspace/aop\_lab/app\_data/.cache/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a

HPSv2 有一个官方的 Python 包 hpsv2。然而，直接使用该包在离线环境下会遇到困难，因为该包内部硬编码了从 Hugging Face 下载权重的逻辑，并且默认将权重存放在 \~/.cache/hpsv2/ 下，而不是用户提供的路径。

为了使用用户提供的特定路径，我们有两种策略：

1. **软链接法**：将用户的路径软链接到 hpsv2 库期望的目录。  
2. **自定义加载法**（推荐）：绕过 hpsv2 库的加载逻辑，直接读取权重文件。

本报告采用**自定义加载法**，因为这样更稳健，不依赖于库的内部实现细节。HPSv2 的核心是一个 OpenCLIP 模型。

### **6.3 本地调用实现代码**

我们需要依赖 open\_clip\_torch 库。

Python

import open\_clip  
import torch  
from PIL import Image  
import os  
import glob

class LocalHPSv2Loader:  
    def \_\_init\_\_(self, snapshot\_path, device="cuda"):  
        self.device \= device  
        print(f" 初始化中，快照路径: {snapshot\_path}")  
          
        \# 1\. 寻找 checkpoint 文件 (.pt)  
        \# HPSv2 的 snapshot 里通常包含 HPS\_v2.1\_compressed.pt 或类似文件  
        pt\_files \= glob.glob(os.path.join(snapshot\_path, "\*.pt"))  
        if not pt\_files:  
            raise FileNotFoundError(f"在路径 {snapshot\_path} 中未找到.pt 权重文件")  
          
        checkpoint\_path \= pt\_files  
        print(f" 锁定权重文件: {checkpoint\_path}")

        \# 2\. 初始化 OpenCLIP 模型结构  
        \# HPSv2 使用的是 ViT-H-14, 预训练于 laion2b\_s32b\_b79k  
        \# 注意：这里 create\_model 可能会尝试下载 config，如果完全离线可能会报错  
        \# 解决方法是手动指定 config 或确保 open\_clip 缓存了该架构配置  
        try:  
            self.model, \_, self.preprocess \= open\_clip.create\_model\_and\_transforms(  
                'ViT-H-14',   
                pretrained=None,  \# 不下载预训练权重，因为我们要加载自己的  
                precision='amp',  
                device=device,  
                force\_quick\_gelu=False,  
                pretrained\_image=False  
            )  
        except Exception as e:  
            print(" OpenCLIP 初始化失败，请检查 open\_clip\_torch 版本。")  
            raise e  
              
        \# 3\. 加载权重  
        print(f" 加载权重...")  
        checkpoint \= torch.load(checkpoint\_path, map\_location=device)  
        \# 检查 checkpoint 结构，通常是 {'state\_dict':...}  
        if 'state\_dict' in checkpoint:  
            self.model.load\_state\_dict(checkpoint\['state\_dict'\])  
        else:  
            self.model.load\_state\_dict(checkpoint)  
              
        self.model.eval()  
        self.tokenizer \= open\_clip.get\_tokenizer('ViT-H-14')  
        print(" 加载完毕。")

    def score(self, image\_input, prompt):  
        if isinstance(image\_input, str):  
            image \= Image.open(image\_input).convert("RGB")  
        else:  
            image \= image\_input

        image\_tensor \= self.preprocess(image).unsqueeze(0).to(self.device)  
        text\_tensor \= self.tokenizer(\[prompt\]).to(self.device)

        with torch.no\_grad():  
            image\_features \= self.model.encode\_image(image\_tensor)  
            text\_features \= self.model.encode\_text(text\_tensor)

            image\_features /= image\_features.norm(dim=-1, keepdim=True)  
            text\_features /= text\_features.norm(dim=-1, keepdim=True)

            hps\_score \= image\_features @ text\_features.T  
              
        return hps\_score.item()

\# 实例化示例  
if \_\_name\_\_ \== "\_\_main\_\_":  
    hpsv2\_path \= "/home/admin/workspace/aop\_lab/app\_data/.cache/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a"  
    hpsv2\_scorer \= LocalHPSv2Loader(hpsv2\_path)

## ---

**7\. BLIP架构与遗留挑战：zai-org/ImageReward**

### **7.1 模型概览与理论基础**

**ImageReward** 基于 BLIP (Bootstrapping Language-Image Pre-training) 架构。与 CLIP 相比，BLIP 在处理复杂的图像描述和多模态交互方面表现更佳。ImageReward 通过在 ImageRewardDB 上训练，学习识别生成图像中的具体缺陷（如肢体扭曲、物体缺失）。

### **7.2 依赖冲突与版本地狱**

用户提到“这个是老模型了，可能环境不兼容”。这是一个非常精准的洞察。

* **冲突点**：ImageReward 依赖旧版的 transformers (约 4.27) 和 diffusers。而后续的 HPSv3 需要极新的 transformers (4.37+) 以支持 Qwen2-VL。  
* **后果**：在同一个 Python 进程中同时 import hpsv3 和 import ImageReward 极大概率会崩溃，或者导致其中一个无法正常加载。

### **7.3 本地路径与加载策略**

路径：/home/admin/workspace/aop\_lab/app\_data/.cache/models--zai-org--ImageReward/snapshots/5736be03b2652728fb87788c9797b0570450ab72

为了在本地加载，我们可以尝试使用 ImageReward 官方包的 .load() 方法，但需要修改其源码或使用特殊的调用方式来指向本地路径，而不是让它去 Hugging Face 下载。

**推荐方案**：如果是为了研究，建议为 ImageReward 单独建立一个 Conda 环境。如果必须集成，则建议将其封装为一个独立的子进程服务。

以下是在兼容环境下的**直接加载代码**（绕过包的自动下载）：

Python

import torch  
import os  
from PIL import Image  
\# 假设已安装 image-reward 包: pip install image-reward  
import ImageReward as RM

class LocalImageRewardLoader:  
    def \_\_init\_\_(self, model\_path, device="cuda"):  
        self.device \= device  
        self.model\_path \= model\_path  
        print(f" 正在尝试从本地加载: {model\_path}")  
          
        \# ImageReward 库的.load() 函数通常接受模型名称  
        \# 如果我们传入绝对路径，库可能会报错，因为它期望的是一个预定义的名称  
        \# 因此，我们需要“欺骗”或者直接实例化其内部类  
          
        try:  
            \# 尝试直接调用 load，部分版本支持本地路径  
            self.model \= RM.load(model\_path, device=device)  
        except Exception as e:  
            print(f" 标准加载失败 ({e})，尝试手动构建模型...")  
            \# 手动构建逻辑 (基于 BLIP)  
            \# 这需要深入 ImageReward 源码结构，以下是通用适配逻辑  
            from ImageReward.ImageReward import ImageReward  
              
            \# 寻找.pt 文件  
            state\_dict\_path \= os.path.join(model\_path, "ImageReward.pt")  
            if not os.path.exists(state\_dict\_path):  
                 \# 搜索目录下所有.pt  
                 import glob  
                 pts \= glob.glob(os.path.join(model\_path, "\*.pt"))  
                 if pts: state\_dict\_path \= pts  
              
            \# 实例化空模型  
            self.model \= ImageReward(device=device)  
            \# 加载权重  
            state\_dict \= torch.load(state\_dict\_path, map\_location=device)  
            self.model.load\_state\_dict(state\_dict, strict=False)  
            self.model.to(device)  
              
        print(" 加载成功。")

    def score(self, image\_input, prompt):  
        \# ImageReward 的 score 函数内部处理了 PIL 打开和预处理  
        \# score \= model.score(prompt, image\_paths)  
        if not isinstance(image\_input, list):  
            image\_inputs \= \[image\_input\]  
        else:  
            image\_inputs \= image\_input  
              
        \# 注意：ImageReward 库通常接受文件路径或 PIL 对象  
        \# 如果是 PIL 对象，确保库版本支持  
        return self.model.score(prompt, image\_inputs)

if \_\_name\_\_ \== "\_\_main\_\_":  
    ir\_path \= "/home/admin/workspace/aop\_lab/app\_data/.cache/models--zai-org--ImageReward/snapshots/5736be03b2652728fb87788c9797b0570450ab72"  
    \# 注意：运行此代码前需确保环境兼容  
    ir\_scorer \= LocalImageRewardLoader(ir\_path)

## ---

**8\. 视觉语言大模型：MizzenAI/HPSv3**

### **8.1 模型概览与理论基础**

**HPSv3** 代表了奖励模型的最新方向——使用**多模态大模型 (LMM/VLM)**。它基于阿里通义千问 Qwen2-VL-7B 架构。与前述输出标量分数的模型不同，HPSv3 可以理解为一个“会看图说话”的 AI，它不仅能给出一个偏好分数，还能通过思维链（Chain of Thought）推理出为什么这张图好或不好。

### **8.2 魔搭社区 (ModelScope) 本地路径处理**

用户指出：“这个是从魔搭社区下载的，本地路径是魔搭社区下载的默认本地路径”。  
通常，ModelScope 的默认缓存路径在 Linux 系统下为：  
\~/.cache/modelscope/hub/MizzenAI/HPSv3  
HPSv3 的官方推理代码 HPSv3RewardInferencer 依赖于 hpsv3 包。我们需要确保该包能够识别我们的本地路径。

### **8.3 依赖与硬件要求**

* **显存**：由于基于 7B 模型，半精度（BF16）加载至少需要 16GB VRAM，推荐 24GB+。  
* **依赖**：pip install hpsv3 flash-attn. 注意 flash-attn 的安装非常依赖 CUDA 版本，本地编译极易失败，建议使用预编译的 whl 包。

### **8.4 本地调用实现代码**

Python

import os  
import torch  
from hpsv3 import HPSv3RewardInferencer

class LocalHPSv3Loader:  
    def \_\_init\_\_(self, local\_path=None, device="cuda"):  
        self.device \= device  
          
        \# 1\. 确定路径  
        if local\_path is None:  
            \# 尝试推断 ModelScope 默认路径  
            home \= os.path.expanduser("\~")  
            possible\_path \= os.path.join(home, ".cache/modelscope/hub/MizzenAI/HPSv3")  
            if os.path.exists(possible\_path):  
                local\_path \= possible\_path  
            else:  
                \# 如果找不到，需要用户显式提供  
                raise FileNotFoundError("未找到 HPSv3 默认路径，请显式指定 local\_path。")  
          
        print(f" 从路径加载: {local\_path}")  
          
        \# 2\. 初始化  
        \# HPSv3RewardInferencer 内部通常会调用 transformers 的加载逻辑  
        \# 我们可以通过 model\_name\_or\_path 参数强制指定本地路径  
        \# 从而避免它去连接 ModelScope 或 HF 下载  
        try:  
            self.inferencer \= HPSv3RewardInferencer(  
                model\_name\_or\_path=local\_path,  
                device=device  
            )  
        except TypeError:  
            \# 如果包版本不支持直接传参，可能需要手动设置属性  
            print(" 构造函数不支持直接传路径，尝试标准初始化...")  
            \# 注意：这里假设 hpsv3 包在检测到本地缓存存在时会优先使用  
            \# 如果它坚持联网，我们可能需要断网运行或设置环境变量 HF\_DATASETS\_OFFLINE=1  
            os.environ \= '1'  
            self.inferencer \= HPSv3RewardInferencer(device=device)

    def score(self, image\_paths, prompts):  
        """  
        HPSv3 专为比较设计，但也支持单图打分（输出 logits）  
        Args:  
            image\_paths: 图片路径列表 \["1.jpg", "2.jpg"\]  
            prompts: 对应的提示词列表  
        """  
        \# inferencer.reward 返回的是包含偏好分数的对象  
        rewards \= self.inferencer.reward(  
            prompts=prompts,  
            image\_paths=image\_paths  
        )  
          
        \# 解析返回值，通常第一个元素是分数  
        scores \= \[r.item() for r in rewards\]  
        return scores

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# 假设路径  
    \# hpsv3\_path \= "/root/.cache/modelscope/hub/MizzenAI/HPSv3"  
    \# hpsv3\_scorer \= LocalHPSv3Loader(hpsv3\_path)  
    pass

## ---

**9\. 综合比较与工程建议**

### **9.1 模型特性对比表**

| 模型 | 核心架构 | 训练数据 | 优势 | 劣势 | 本地化难度 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **CLIP-L** | ViT-L/14 | WIT-400M | 基础、快速、无需微调 | 对齐性差，不懂审美 | 低 |
| **Aesthetic** | Linear Probe | LAION-Aesthetics | 极快、专注纯美学 | 忽略文本，功能单一 | 低 |
| **PickScore** | ViT-H/14 | Pick-a-Pic | 鲁棒性强，人类偏好对齐 | 模型较大 (Huge) | 中 |
| **HPSv2** | ViT-H/14 | HPDv2 | 学术界基准，泛化好 | 需绕过包的下载逻辑 | 中 |
| **ImageReward** | BLIP | ImageRewardDB | 懂细节缺陷 (畸变等) | **依赖冲突严重**，环境老旧 | 高 |
| **HPSv3** | Qwen2-VL | HPDv3 (1M+) | **SOTA**，懂推理，最精准 | 显存巨大，推理慢 | 高 |

### **9.2 环境隔离策略 (Environment Isolation)**

鉴于 **ImageReward** 与 **HPSv3** 的依赖冲突，强烈建议不要尝试在一个 Python 环境中同时加载所有模型。

**推荐架构：**

1. **环境 A (Legacy)**: Python 3.8, transformers==4.27, diffusers==0.16. 专门运行 **ImageReward**。  
2. **环境 B (Modern)**: Python 3.10+, transformers\>=4.37, flash-attn. 运行 **HPSv3, HPSv2, PickScore, CLIP**.
