import cv2
import os
import numpy as np
import pydicom

PASTA_POSITIVA = "Pneumo"
PASTA_NEGATIVA = "NoPneumo"
ARQUIVO_SAIDA = "Resultado.txt"

def carregar_base_dados():
    base_dados = []
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    diretorios = [
        {"caminho": os.path.join(diretorio_atual, PASTA_POSITIVA), "classe": 1, "nome": "Pneumothorax"},
        {"caminho": os.path.join(diretorio_atual, PASTA_NEGATIVA), "classe": 0, "nome": "Negativo"}
    ]


    for pasta in diretorios:
        caminho = pasta["caminho"]
        if not os.path.exists(caminho):
            continue

        contador = 0
        for arq in os.listdir(caminho):
            if arq.lower().endswith('.dcm'):
                caminho_completo = os.path.join(caminho, arq)
                try:
                    ds = pydicom.dcmread(caminho_completo)
                    img = ds.pixel_array.astype(float)
                    
                    # Normalização para 0-255
                    img = (np.maximum(img, 0) / img.max()) * 255.0
                    img = np.uint8(img)
                    
                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    
                    base_dados.append({"hist": hist, "classe": pasta["classe"]})
                    contador += 1
                except Exception as e:
                    print(f"Erro")
        
    return base_dados

def classificar():
    dados = carregar_base_dados()
    total = len(dados)
    
    if total == 0:
        return

    metodos = {
        "CORRELATION": {"id": cv2.HISTCMP_CORREL, "inv": False},
        "CHI-SQUARE": {"id": cv2.HISTCMP_CHISQR, "inv": True},
        "INTERSECTION": {"id": cv2.HISTCMP_INTERSECT, "inv": False},
        "BHATTACHARYYA": {"id": cv2.HISTCMP_BHATTACHARYYA, "inv": True}
    }

    relatorio = "RELATÓRIO DE CLASSIFICAÇÃO - HISTOGRAMAS DICOM\n"
    relatorio += "="*50 + "\n\n"


    for nome, config in metodos.items():
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for i in range(total):
            teste = dados[i]
            melhor_score = float('inf') if config["inv"] else -float('inf')
            predita = -1

            for j in range(total):
                if i == j: continue
                score = cv2.compareHist(teste["hist"], dados[j]["hist"], config["id"])
                
                if config["inv"]:
                    if score < melhor_score:
                        melhor_score = score
                        predita = dados[j]["classe"]
                else:
                    if score > melhor_score:
                        melhor_score = score
                        predita = dados[j]["classe"]

            if teste["classe"] == 1:
                if predita == 1: tp += 1
                else: fn += 1
            else:
                if predita == 0: tn += 1
                else: fp += 1

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        espec = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Montando o texto para este método
        bloco = f"MÉTODO: {nome}\n"
        bloco += f"{'-'*20}\n"
        bloco += f"Matriz de Confusão:\n"
        bloco += f"            Prev P | Prev N\n"
        bloco += f"Real P (Pos):  {tp:2d}   |   {fn:2d}\n"
        bloco += f"Real N (Neg):  {fp:2d}   |   {tn:2d}\n"
        bloco += f"Sensibilidade: {sens:.2%}\n"
        bloco += f"Especificidade: {espec:.2%}\n"
        bloco += "\n" + "="*50 + "\n\n"
        relatorio += bloco

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        f.write(relatorio)


if __name__ == "__main__":
    classificar()
