import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def graficar():
    archivos = glob.glob("results_rank*.csv")
    archivos.sort()
    
    if not archivos:
        print("No hay archivos CSV.")
        return

    nodos = []
    tiempos_off = []
    tiempos_auto = []
    
    # Recolectar datos
    for archivo in archivos:
        df = pd.read_csv(archivo)
        df.columns = [c.strip() for c in df.columns]
        
        rank = df['Rank'].iloc[0]
        nombre = f"Nodo {rank}"
        
        t_off = df[df['Balanced_Mode'] == 'OFF']['Time_Seconds'].mean()
        t_auto = df[df['Balanced_Mode'] != 'OFF']['Time_Seconds'].mean()
        
        nodos.append(nombre)
        tiempos_off.append(t_off)
        tiempos_auto.append(t_auto)

    x = np.arange(len(nodos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(
        x - width/2,
        tiempos_off,
        width,
        label='OFF (Sin Balanceo)',
        color='#f4a261',     # 🟠 Naranja
        edgecolor='black'
    )

    rects2 = ax.bar(
        x + width/2,
        tiempos_auto,
        width,
        label='AUTO (Con Balanceo)',
        color='#2a9df4',     # 🔵 Azul
        edgecolor='black'
    )

    ax.set_ylabel('Tiempo por Época (segundos)', fontsize=12)
    ax.set_title(
        'Impacto del Balanceo de Carga en Entrenamiento Distribuido',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(nodos, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas de valor
    ax.bar_label(rects1, padding=3, fmt='%.2fs', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.2fs', fontsize=10)

    plt.tight_layout()
    plt.savefig('grafico_comparativo.png', dpi=300)
    print("✅ Gráfico guardado como: grafico_comparativo.png")
    plt.show()

if __name__ == "__main__":
    graficar()
