import pandas as pd
import glob
import os
import sys

def analizar_resultados():
    # Busca todos los CSV que empiecen con "results_rank"
    archivos = glob.glob("results_rank*.csv")
    archivos.sort()

    if not archivos:
        print("❌ No encontré archivos .csv en esta carpeta.")
        print("   Asegúrate de haber hecho los 'scp' desde las otras máquinas.")
        return

    # Encabezado de la tabla
    print("\n" + "="*85)
    print(f"{'ARCHIVO / NODO':<20} | {'MODO':<10} | {'TIEMPO (s)':<12} | {'BATCH':<8} | {'MEJORA'}")
    print("="*85)

    promedio_mejora = 0
    count = 0

    for archivo in archivos:
        try:
            df = pd.read_csv(archivo)
            
            # Limpiar espacios en blanco en los nombres de columnas si los hubiera
            df.columns = [c.strip() for c in df.columns]

            # 1. Analizar Baseline (OFF)
            df_off = df[df['Balanced_Mode'] == 'OFF']
            time_off = df_off['Time_Seconds'].mean() if not df_off.empty else 0
            
            # 2. Analizar Optimizado (AUTO, ON, etc.)
            # Tomamos todo lo que NO sea OFF
            df_on = df[df['Balanced_Mode'] != 'OFF']
            time_on = df_on['Time_Seconds'].mean() if not df_on.empty else 0
            batch_on = int(df_on['Batch_Size'].mean()) if not df_on.empty else 0

            # 3. Calcular porcentaje de mejora
            if time_off > 0 and time_on > 0:
                mejora = ((time_off - time_on) / time_off) * 100
                txt_mejora = f"🚀 {mejora:.2f}% MÁS RÁPIDO"
                promedio_mejora += mejora
                count += 1
            elif time_on == 0:
                txt_mejora = "⏳ Falta correr AUTO"
            else:
                txt_mejora = "N/A"

            # Imprimir fila del nodo
            nodo_nombre = os.path.basename(archivo).replace(".csv", "")
            print(f"🔹 {nodo_nombre}")
            print(f"{'   (Sin Balanceo)':<20} | {'OFF':<10} | {time_off:<12.4f} | {'1000':<8} | -")
            print(f"{'   (Con Balanceo)':<20} | {'AUTO':<10} | {time_on:<12.4f} | {str(batch_on):<8} | {txt_mejora}")
            print("-" * 85)

        except Exception as e:
            print(f"⚠️ Error leyendo {archivo}: {e}")

    if count > 0:
        total = promedio_mejora / count
        print(f"\n✨ RESULTADO FINAL DEL CLÚSTER: {total:.2f}% de aceleración promedio. ✨")
        print("="*85 + "\n")

if __name__ == "__main__":
    analizar_resultados()