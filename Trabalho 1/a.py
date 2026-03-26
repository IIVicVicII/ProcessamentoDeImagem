import re
import os
import glob

def processar_experimento_microscopia(diretorio_alvo):
    # 1. Configuração do padrão de busca e Regex
    # O asterisco (*) captura qualquer tamanho de imagem (512, 608, 800)
    padrao_arquivos = os.path.join(diretorio_alvo, "*_yolov4_5000_dados_celulas.txt")
    padrao_regex = r"conf_thresh = ([\d\.]+), TP = (\d+), FP = (\d+), FN = (\d+), average IoU = ([\d\.]+) %"
    
    # 2. Localizar os ficheiros na pasta
    lista_arquivos = glob.glob(padrao_arquivos)
    
    if not lista_arquivos:
        print(f"❌ Erro: Nenhum ficheiro encontrado em '{diretorio_alvo}'")
        print("Verifique se o caminho está correto e se os ficheiros foram extraídos do ZIP.")
        return
    
    resultados_totais = {}

    # 3. Processar cada ficheiro encontrado
    for caminho_completo in lista_arquivos:
        nome_curto = os.path.basename(caminho_completo)
        print(f"🔍 Analisando: {nome_curto}...")
        
        dados_extraidos = []
        
        try:
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                for linha in f:
                    match = re.search(padrao_regex, linha)
                    if match:
                        # Criar dicionário com os valores convertidos
                        entrada = {
                            "threshold": float(match.group(1)),
                            "TP": int(match.group(2)),
                            "FP": int(match.group(3)),
                            "FN": int(match.group(4)),
                            "IoU": float(match.group(5))
                        }
                        dados_extraidos.append(entrada)
            
            resultados_totais[nome_curto] = dados_extraidos
            
        except Exception as e:
            print(f"⚠️ Erro ao ler o ficheiro {nome_curto}: {e}")

    # 4. Exibição de um resumo dos resultados
    exibir_resumo(resultados_totais)
    return resultados_totais

def exibir_resumo(dados):
    print("\n" + "="*50)
    print("RESUMO DA EXTRAÇÃO DE DADOS")
    print("="*50)
    for arquivo, analises in dados.items():
        print(f"📌 Ficheiro: {arquivo}")
        print(f"   -> {len(analises)} níveis de threshold processados.")
        if analises:
            # Mostra apenas o primeiro e o último para conferência
            prim = analises[0]
            ult = analises[-1]
            print(f"   -> Range: Thresh {prim['threshold']} (TP:{prim['TP']}) até {ult['threshold']} (TP:{ult['TP']})")
    print("="*50)

# --- CONFIGURAÇÃO ---
# Se os ficheiros estiverem na mesma pasta do script, mantém "."
# Se estiverem noutra pasta, coloca o caminho entre aspas: r"C:\Pasta\Dados"
meu_diretorio = "./arquivos" 

# Iniciar o programa
processar_experimento_microscopia(meu_diretorio)