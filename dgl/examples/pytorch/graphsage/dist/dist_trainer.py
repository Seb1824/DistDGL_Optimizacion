import argparse
import socket
import time
from contextlib import contextmanager
import csv
import os
import subprocess

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

# INICIO BLOQUE WIFI AUTO-DETECT
try:
    cmd = "ls /sys/class/net | grep wlp | head -n 1"
    wifi_iface = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    if wifi_iface:
        os.environ['GLOO_SOCKET_IFNAME'] = wifi_iface
        os.environ['TP_SOCKET_IFNAME'] = wifi_iface
        print(f"\n🔥 RED DETECTADA: Usando tarjeta Wi-Fi: {wifi_iface} 🔥\n")
except Exception as e:
    print(f"Advertencia: No se pudo detectar Wi-Fi automatico. Error: {e}")
# --- FIN BLOQUE WIFI ---

# --- FUNCIONES AUXILIARES ---
def log_to_csv(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Cabecera del CSV actualizada con GPU_Score
            writer.writerow(['Timestamp', 'Rank', 'Epoch', 'Time_Seconds', 'Loss', 'Train_Acc', 'Balanced_Mode', 'Batch_Size', 'GPU_Score'])
        writer.writerow(data)

def measure_gpu_capability(device, duration=2.0):
    """
    Ejecuta multiplicaciones de matrices durante 'duration' segundos
    para medir la potencia bruta del hardware.
    Retorna: ops_per_second (Score)
    """
    # Tensores grandes para estresar la GPU/CPU
    size = 2000
    x = th.randn(size, size, device=device)
    y = th.randn(size, size, device=device)
    
    # Warm-up (calentamiento)
    print(f"   ⚙️ Calibrando hardware en {device}...")
    for _ in range(3):
        _ = th.mm(x, y)
    if device.type == 'cuda':
        th.cuda.synchronize()

    start = time.time()
    ops = 0
    while (time.time() - start) < duration:
        _ = th.mm(x, y)
        ops += 1
    
    if device.type == 'cuda':
        th.cuda.synchronize()
    
    real_time = time.time() - start
    score = ops / real_time
    return score

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    batch_inputs = (
        g.ndata["feat"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["labels"][seeds].to(device)
    return batch_inputs, batch_labels

class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    @contextmanager
    def join(self):
        yield

def compute_acc(pred, labels):
    labels = labels.long()
    if len(labels.shape) > 1:
        labels = labels.squeeze(1)
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def run(args, device, data):
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    shuffle = True

    # AUTO-TUNING: CALCULO GENERALIZADO DE BATCH SIZE
    current_bs = args.batch_size
    gpu_score = 0
    mode_status = "BASELINE (Static)"

    if args.enable_balancing:
        mode_status = "AUTO-TUNED (Dynamic)"
        print(f"\n⚡ [AUTO-TUNE] Iniciando benchmark de hardware en Rank {g.rank()}...")
        
        # 1. Medimos la potencia real
        gpu_score = measure_gpu_capability(device)
        print(f"   📊 GPU Score obtenido: {gpu_score:.2f} Ops/sec")

        # 2. Score de Referencia (Ajustable segun sus resultados previos)
        #    Usamos 160.0 como base. Si tu GPU saca mas, recibe mas carga.
        REFERENCE_SCORE = 160.0 

        # 3. Calculo del factor
        scale_factor = gpu_score / REFERENCE_SCORE
        
        # 4. Limites de seguridad (Min 50% carga, Max 150% carga)
        scale_factor = max(0.5, min(scale_factor, 1.5))
        
        # 5. Aplicar nuevo Batch Size
        new_bs = int(args.batch_size * scale_factor)
        
        print(f"   ⚖️ Factor calculado: {scale_factor:.2f}x")
        print(f"   🔄 Batch Size ajustado: {args.batch_size} -> {new_bs}")
        args.batch_size = new_bs
    else:
        print(f"\n🐢 [BASELINE] Usando Batch Size fijo: {args.batch_size}")

    # -------------------------------------------------------------

    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    
    model = DistSAGE(
        in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout
    )
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )
            
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n--- Iniciando Entrenamiento: {mode_status} ---\n")
    
    for epoch in range(args.num_epochs):
        tic = time.time()

        total_loss = 0
        total_acc = 0
        num_steps = 0

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                
                batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, "cpu")
                batch_labels = batch_labels.long()
                if len(batch_labels.shape) > 1:
                    batch_labels = batch_labels.squeeze(1)

                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metricas
                acc = compute_acc(batch_pred, batch_labels)
                total_loss += loss.item()
                total_acc += acc.item()
                num_steps += 1

                if step % args.log_every == 0:
                    gpu_mem = th.cuda.max_memory_allocated() / 1e6 if th.cuda.is_available() else 0
                    print(f"Part {g.rank()} | Ep {epoch:02d} | St {step:03d} | Loss {loss.item():.4f} | Acc {acc.item():.4f} | GPU {gpu_mem:.1f}MB")

        toc = time.time()
        epoch_time = toc - tic
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        avg_acc = total_acc / num_steps if num_steps > 0 else 0

        print(f"✅ Part {g.rank()} | Epoch {epoch} Finalizada | Tiempo: {epoch_time:.4f}s")
        
        # --- GUARDAR DATOS EN CSV ---
        log_data = [
            time.strftime("%Y-%m-%d %H:%M:%S"), # Timestamp
            g.rank(),                           # Quien soy
            epoch,                              # Epoca actual
            round(epoch_time, 4),               # Tiempo que tardo
            round(avg_loss, 4),                 # Loss promedio
            round(avg_acc, 4),                  # Acc promedio
            "AUTO" if args.enable_balancing else "OFF", # Modo
            args.batch_size,                    # Carga usada
            round(gpu_score, 2)                 # Score de la maquina
        ]
        log_filename = f"results_rank{g.rank()}.csv"
        log_to_csv(log_filename, log_data)
        # ----------------------------

def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    pb = g.get_partition_book()
    
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"], pb, force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata["val_mask"], pb, force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata["test_mask"], pb, force_even=True)
    
    if th.cuda.is_available():
        device = th.device("cuda:0")
        print(f"🚀 {socket.gethostname()} [Rank {g.rank()}]: GPU {th.cuda.get_device_name(0)}")
    else:
        device = th.device("cpu")
    
    n_classes = args.n_classes
    if 'arxiv' in args.graph_name: n_classes = 40
    elif 'products' in args.graph_name: n_classes = 47
    
    in_feats = g.ndata["feat"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    
    run(args, device, data)
    print("Entrenamiento Finalizado Correctamente")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN Distribuido")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration")
    parser.add_argument("--part_config", type=str, help="The path to the partition config file")
    parser.add_argument("--n_classes", type=int, default=0)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--pad-data", default=False, action="store_true")
    parser.add_argument("--net_type", type=str, default="socket")
    
    parser.add_argument("--enable_balancing", action="store_true", help="Activar balanceo de carga para GPUs heterogeneas")
    
    args = parser.parse_args()
    main(args)
    