import re
import os

def processar_arquivos_yolo(lista_arquivos):
    # Dicionário para guardar todos os resultados
    resultados_finais = {}

    # Expressão regular para encontrar os valores numéricos na linha
    # Ela procura por: threshold, TP, FP, FN e IoU
    padrao_regex = r"conf_thresh = ([\d\.]+), TP = (\d+), FP = (\d+), FN = (\d+), average IoU = ([\d\.]+) %"

    for nome_arquivo in lista_arquivos:
        if not os.path.exists(nome_arquivo):
            print(f"⚠️ Arquivo {nome_arquivo} não encontrado.")
            continue
            
        print(f"📂 Processando: {nome_arquivo}...")
        dados_do_arquivo = []

        with open(nome_arquivo, 'r') as file:
            for linha in file:
                # Procura o padrão na linha
                match = re.search(padrao_regex, linha)
                
                if match:
                    # Extrai os grupos encontrados e converte para os tipos corretos
                    dados = {
                        "threshold": float(match.group(1)),
                        "TP": int(match.group(2)),
                        "FP": int(match.group(3)),
                        "FN": int(match.group(4)),
                        "IoU_percent": float(match.group(5))
                    }
                    dados_do_arquivo.append(dados)
        
        # Guarda os dados usando o nome do arquivo como chave
        resultados_finais[nome_arquivo] = dados_do_arquivo

    return resultados_finais

# --- Execução do Programa ---

# 1. Lista dos arquivos que o seu supervisor entregou
arquivos_para_ler = [
    "512_Yolov4_5000_dados_celulas.txt",
    "608_Yolov4_5000_dados_celulas.txt",
    "800_Yolov4_5000_dados_celulas.txt"
]

# 2. Chamar a função de processamento
dados_extraidos = processar_arquivos_yolo(arquivos_para_ler)

# 3. Exemplo de como visualizar os dados extraídos
for arquivo, analises in dados_extraidos.items():
    print(f"\n--- Resumo para {arquivo} ---")
    # Mostra apenas os 3 primeiros para não encher a tela
    for item in analises[:3]:
        print(f"Thresh: {item['threshold']} | TP: {item['TP']} | IoU: {item['IoU_percent']}%")
    print(f"Total de thresholds lidos: {len(analises)}")