import re
import os
import glob
import matplotlib.pyplot as plt

def calcular_metricas(dados):
    for d in dados:
        tp, fp, fn = d['TP'], d['FP'], d['FN']
        
        # Calculo da precisao 
        d['precision'] = tp / (tp + fp) 
        
        # Calculo da sensibilidade
        d['recall'] = tp / (tp + fn) 
        
        p, r = d['precision'], d['recall']
        
        # Calculo de F1
        d['f1'] = 2 * (p * r) / (p + r)

    return dados

def processar_e_analisar(diretorio):
    # Processamento dos arquivos e coleta dos dados que nos interessa
    padrao_regex = r"conf_thresh = ([\d\.]+), TP = (\d+), FP = (\d+), FN = (\d+), average IoU = ([\d\.]+) %"
    arquivos = glob.glob(os.path.join(diretorio, "*_yolov4_5000_dados_celulas.txt"))
    
    if not arquivos:
        print(f"Erro ao procurar arquivos, coloque os arquivos desejados no diretório 'arquivos'")
        return

    plt.figure(figsize=(10, 7))
    todos_os_resultados = []

    for arq_path in arquivos:
        nome = os.path.basename(arq_path)
        tamanho = nome.split('_')[0]
        
        dados_arquivo = []
        with open(arq_path, 'r', encoding='utf-8') as f:
            for linha in f:
                m = re.search(padrao_regex, linha)
                if m:
                    dados_arquivo.append({
                        "thresh": float(m.group(1)), 
                        "TP": int(m.group(2)), 
                        "FP": int(m.group(3)), 
                        "FN": int(m.group(4)),
                        "tamanho": tamanho
                    })
        
        if not dados_arquivo: continue
        
        # Calcular metricas e adicionar a lista global para o Ranking
        dados_arquivo = calcular_metricas(dados_arquivo)
        todos_os_resultados.extend(dados_arquivo)
        
        # Dados para o grafico
        recalls = [d['recall'] for d in dados_arquivo]
        precisions = [d['precision'] for d in dados_arquivo]
        
        plt.plot(recalls, precisions, '-o', markersize=3, label=f"Res: {tamanho}")

    ranking_f1 = sorted(todos_os_resultados, key=lambda x: x['f1'], reverse=True)
    top_3 = ranking_f1[:3]

    # Gerar o grafico
    plt.xlabel('Recall (Sensibilidade)')
    plt.ylabel('Precision (Precisão)')
    plt.title('Curva Precision-Recall')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('curva_precision_recall.png')

    # Impressao no terminal dos resultados
    print("Maiores F1:\n")
    for i, res in enumerate(top_3, 1):
        print(f"{i}º: Tamanho {res['tamanho']}px | Threshold: {res['thresh']}")
        print(f"   Metricas: F1: {res['f1']:.4f} | Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f}\n")
    

# Executar
processar_e_analisar("./arquivos")