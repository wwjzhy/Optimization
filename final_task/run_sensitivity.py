import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.float32 else G
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov: g = g.add(buf, alpha=momentum)
                else: g = buf
                if p.ndim >= 2:
                    g_2d = g.view(g.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                    g_update = g_ortho.view_as(p)
                    p.data.add_(g_update, alpha=-lr * max(1, g_2d.size(0)/g_2d.size(1))**0.5)
                else:
                    p.data.add_(g, alpha=-lr)

class DynamicCNN(nn.Module):
    def __init__(self, num_layers=5): # 固定 5 层
        super(DynamicCNN, self).__init__()
        self.features = nn.Sequential()
        in_c, out_c = 3, 32
        for i in range(num_layers):
            self.features.add_module(f'c{i}', nn.Conv2d(in_c, out_c, 3, 1, 1))
            self.features.add_module(f'r{i}', nn.ReLU())
            self.features.add_module(f'bn{i}', nn.BatchNorm2d(out_c))
            if i < 3: self.features.add_module(f'p{i}', nn.MaxPool2d(2))
            in_c = out_c
            if out_c < 128: out_c *= 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_c, 10)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.features(x)), 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

EPOCHS_PER_RUN = 8  # 每个图跑8个epoch，足够看清趋势
LRS_TO_TEST = [0.05, 0.01, 0.005, 0.001]
OPTIMIZERS = ['Muon', 'Adam', 'RMSprop', 'SGD']

STYLES = {
    'Muon':    {'c': '#d62728', 'ls': '-', 'lw': 2.5}, # 红，粗实线
    'Adam':    {'c': '#1f77b4', 'ls': '--', 'lw': 1.5}, # 蓝，虚线
    'RMSprop': {'c': '#ff7f0e', 'ls': '-.', 'lw': 1.5}, # 橙，点划线
    'SGD':     {'c': '#2ca02c', 'ls': ':',  'lw': 1.5}  # 绿，点线
}

def train_one_epoch(opt_name, lr):
    model = DynamicCNN(num_layers=5).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 设定优化器
    if opt_name == 'Muon':
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.ndim < 2]
        # 关键：Muon 部分使用测试 LR，背景 AdamW 固定为 0.001 以控制变量
        # 或者为了更严格的测试，背景 AdamW 也可以设为 lr/10。这里我们选择固定。
        opt = Muon(muon_params, lr=lr, momentum=0.95)
        opt_aux = optim.AdamW(adam_params, lr=0.001, weight_decay=0.01)
        optimizers = [opt, opt_aux]
    elif opt_name == 'Adam':
        optimizers = [optim.Adam(model.parameters(), lr=lr)]
    elif opt_name == 'RMSprop':
        optimizers = [optim.RMSprop(model.parameters(), lr=lr)]
    elif opt_name == 'SGD':
        optimizers = [optim.SGD(model.parameters(), lr=lr, momentum=0.9)]
    
    loss_curve = []
    
    print(f"   > Training {opt_name}...", end="", flush=True)
    
    for epoch in range(EPOCHS_PER_RUN):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            for o in optimizers: o.zero_grad()
            
            out = model(x)
            loss = criterion(out, y)
            

            if torch.isnan(loss) or loss.item() > 100:
                epoch_loss += 5.0 * len(trainloader) 
                break 
            
            loss.backward()
            for o in optimizers: o.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / max(batches, 1)
        if math.isnan(avg_loss) or avg_loss > 5.0: 
            avg_loss = 2.5 
        loss_curve.append(avg_loss)
        
    print(" Done.")
    return loss_curve

print(f"Starting Multi-Chart Generation for LRs: {LRS_TO_TEST}")

for lr in LRS_TO_TEST:
    print(f"\n=== Generating Plot for Learning Rate {lr} ===")
    plt.figure(figsize=(8, 5))
    
    for opt_name in OPTIMIZERS:
        losses = train_one_epoch(opt_name, lr)
        plt.plot(range(1, EPOCHS_PER_RUN+1), losses, 
                 label=opt_name, 
                 color=STYLES[opt_name]['c'], 
                 linestyle=STYLES[opt_name]['ls'],
                 linewidth=STYLES[opt_name]['lw'])

    plt.title(f'Training Loss Dynamics @ Learning Rate = {lr}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.ylim(0, 2.5) 
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    fname = f"loss_curve_lr_{lr}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")

print("\nAll plots generated successfully.")