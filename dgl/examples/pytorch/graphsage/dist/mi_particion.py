import dgl
import torch
import os
from ogb.nodeproppred import DglNodePropPredDataset

def run():
    # 1. Descargar y cargar el dataset OGB Products
    print("⏳ Descargando dataset 'ogbn-products' (esto puede tardar)...")
    dataset = DglNodePropPredDataset(name='ogbn-products', root='./dataset')
    
    # El dataset devuelve el grafo y las etiquetas por separado
    g, labels = dataset[0]
    
    # Añadimos las etiquetas al grafo para que se repartan a los workers
    g.ndata['labels'] = labels
    
    print(f"✅ Grafo cargado: {g.num_nodes()} nodos, {g.num_edges()} aristas.")

    # 2. Configurar particionado
    num_parts = 3  # Giomar, Anlu, Melany
    out_path = './data_part'
    
    # 3. Ejecutar particionado
    print(f"✂️  Particionando el grafo en {num_parts} pedazos...")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    dgl.distributed.partition_graph(
        g,
        'ogbn-products',
        num_parts,
        out_path,
        part_method='random',  # <--- AGREGA ESTA LÍNEA (No olvides la coma al final)
        balance_edges=True
    )
    
    print(f"🎉 ¡ÉXITO! Datos guardados en {out_path}")

if __name__ == '__main__':
    run()