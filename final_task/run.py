import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Muon 核心的正交化算法"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.float32 else G
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    """Muon 优化器类"""
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
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                # Muon Update for >=2D params (Conv2d weights are 4D, Linear are 2D)
                if p.ndim >= 2:
                    g_2d = g.view(g.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                    g_update = g_ortho.view_as(p)
                    p.data.add_(g_update, alpha=-lr * max(1, g_2d.size(0)/g_2d.size(1))**0.5)
                else:
                    p.data.add_(g, alpha=-lr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 128
EPOCHS = 10 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Preparing Data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class DynamicCNN(nn.Module):
    def __init__(self, num_layers=3):
        super(DynamicCNN, self).__init__()
        self.features = nn.Sequential()
        
        in_channels = 3
        out_channels = 32
        
        for i in range(num_layers):
            self.features.add_module(f'conv{i+1}', nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.features.add_module(f'relu{i+1}', nn.ReLU())
            self.features.add_module(f'bn{i+1}', nn.BatchNorm2d(out_channels))
            
            if i < 3:
                self.features.add_module(f'pool{i+1}', nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
            if out_channels < 128:
                out_channels *= 2
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(num_layers, optimizer_name):
    model = DynamicCNN(num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizers = []
    if optimizer_name == 'Muon':
        muon_params, adamw_params = [], []
        for p in model.parameters():
            if p.ndim >= 2: muon_params.append(p)
            else: adamw_params.append(p)
        optimizers.append(Muon(muon_params, lr=0.02, momentum=0.95))
        optimizers.append(optim.AdamW(adamw_params, lr=0.001, weight_decay=0.01))
    elif optimizer_name == 'Adam':
        optimizers.append(optim.Adam(model.parameters(), lr=0.001))
    elif optimizer_name == 'RMSprop':
        optimizers.append(optim.RMSprop(model.parameters(), lr=0.001))
    elif optimizer_name == 'SGD':
        optimizers.append(optim.SGD(model.parameters(), lr=0.01, momentum=0.9))
    
    loss_history = []
    print(f"   |-- Running {optimizer_name}...", end="", flush=True)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            for opt in optimizers: opt.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            for opt in optimizers: opt.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        
    final_acc = evaluate_accuracy(model, testloader)
    print(f" Done. Final Acc: {final_acc:.2f}%")
    return loss_history, final_acc

target_layers = [3, 4, 5, 6]
optimizer_list = ['Adam', 'RMSprop', 'SGD', 'Muon']
all_results = {} 

styles = {
    'Adam':    {'color': '#1f77b4', 'style': '-'},
    'RMSprop': {'color': '#ff7f0e', 'style': '--'},
    'SGD':     {'color': '#2ca02c', 'style': '-.'},
    'Muon':    {'color': '#d62728', 'style': '-'}
}

print(f"\nStarting Batch Experiments for Depths: {target_layers}")
print("="*60)

for depth in target_layers:
    print(f"\n>>> Experiment Set: Depth {depth} Layers")
    
    depth_results = {}
    depth_acc = {}
    
    for opt in optimizer_list:
        loss, acc = run_experiment(depth, opt)
        depth_results[opt] = loss
        depth_acc[opt] = acc
    
    all_results[depth] = depth_acc
    
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    
    for opt in optimizer_list:
        plt.plot(epochs_range, depth_results[opt], 
                 label=f"{opt}", 
                 color=styles[opt]['color'], 
                 linestyle=styles[opt]['style'],
                 linewidth=2.5 if opt == 'Muon' else 1.5)

    plt.title(f'Loss Convergence - CNN Depth: {depth} Layers', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    
    img_name = f"result_depth_{depth}_layers.png"
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f">>> Saved Plot: {img_name}")

txt_filename = "final_report_all_depths.txt"
with open(txt_filename, "w", encoding="utf-8") as f:
    f.write("Muon vs Others - Depth Comparison Report\n")
    f.write("========================================\n\n")
    
    for depth in target_layers:
        f.write(f"--- Model Depth: {depth} Layers ---\n")
        sorted_acc = sorted(all_results[depth].items(), key=lambda x: x[1], reverse=True)
        for opt, acc in sorted_acc:
            f.write(f"{opt:<10} : {acc:.2f}%\n")
        f.write("\n")

print("\n" + "="*60)
print(f"All experiments finished.")
print(f"Images saved: result_depth_X_layers.png")
print(f"Report saved: {txt_filename}")