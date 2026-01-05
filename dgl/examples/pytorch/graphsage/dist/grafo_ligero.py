import dgl
import torch
import os

def run():
    print("🏗️ Generando grafo sintético ligero...")
    # Crear un grafo aleatorio pequeño (10,000 nodos, 50,000 aristas)
    num_nodes = 10000
    num_edges = 50000
    g = dgl.rand_graph(num_nodes, num_edges)
    
    # Crear datos falsos (features y etiquetas)
    # 128 características por nodo (simulando datos reales)
    g.ndata['feat'] = torch.randn(num_nodes, 128)
    # 10 clases posibles
    g.ndata['labels'] = torch.randint(0, 10, (num_nodes,))
    
    # Crear máscaras de entrenamiento/validación/prueba
    n = num_nodes
    g.ndata['train_mask'] = torch.zeros(n, dtype=torch.bool)
    g.ndata['val_mask'] = torch.zeros(n, dtype=torch.bool)
    g.ndata['test_mask'] = torch.zeros(n, dtype=torch.bool)
    
    g.ndata['train_mask'][:int(n*0.6)] = True
    g.ndata['val_mask'][int(n*0.6):int(n*0.8)] = True
    g.ndata['test_mask'][int(n*0.8):] = True

    print(f"✅ Grafo creado: {g.num_nodes()} nodos. Listo para partir.")

    # Particionar
    num_parts = 3
    out_path = './data_part'
    
    print(f"✂️  Particionando en {num_parts} pedazos (método random)...")
    dgl.distributed.partition_graph(
        g,
        'grafo_prueba', # Nombre del grafo
        num_parts,
        out_path,
        part_method='random',
        balance_edges=True
    )
    
    print(f"🎉 ¡ÉXITO! Datos guardados en {out_path}")

if __name__ == '__main__':
    run()

    