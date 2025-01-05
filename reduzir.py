import os
import pandas as pd

def dividir_csv_por_tamanho(arquivo_entrada, pasta_saida, tamanho_max_mb=25, encoding='utf-8', contador_inicial=1):
    # Garantir que a pasta de saída existe
    os.makedirs(pasta_saida, exist_ok=True)

    # Lê o arquivo inteiro em um DataFrame
    df = pd.read_csv(arquivo_entrada, encoding=encoding)
    
    # Inicializar variáveis
    tamanho_max_bytes = tamanho_max_mb * 1024 * 1024  # Converter MB para bytes
    inicio = 0
    parte = contador_inicial  # Iniciar com o contador global

    while inicio < len(df):
        # Ajusta o intervalo a ser salvo
        chunk = df.iloc[inicio:inicio + 1000]  # Tenta dividir em pedaços de 1000 linhas inicialmente
        while True:
            # Salva temporariamente para verificar o tamanho
            arquivo_temp = os.path.join(pasta_saida, f"temp_parte_{parte}.csv")
            chunk.to_csv(arquivo_temp, index=False, encoding=encoding)
            if os.path.getsize(arquivo_temp) > tamanho_max_bytes:
                # Se o arquivo for maior que o tamanho permitido, reduz o chunk
                chunk = df.iloc[inicio:inicio + len(chunk) - 100]  # Reduz o tamanho em 100 linhas
            else:
                # Tamanho está dentro do limite, renomeia o arquivo e avança
                arquivo_saida = os.path.join(pasta_saida, f"parte_{parte}.csv")
                os.rename(arquivo_temp, arquivo_saida)
                print(f"Arquivo gerado: {arquivo_saida}")
                inicio += len(chunk)
                parte += 1
                break

    print(f"Divisão concluída para o arquivo: {arquivo_entrada}")
    return parte  # Retorna o próximo número disponível

# Processar os arquivos na pasta
def processar_pasta(diretorio_arquivos, pasta_saida, tamanho_max_mb=25, encoding='utf-8'):
    if not os.path.exists(diretorio_arquivos):
        print(f"Diretório de entrada não encontrado: {diretorio_arquivos}")
        return

    arquivos_csv = [
        os.path.join(diretorio_arquivos, arquivo)
        for arquivo in os.listdir(diretorio_arquivos)
        if arquivo.endswith('.csv')
    ]

    if not arquivos_csv:
        print(f"Nenhum arquivo CSV encontrado no diretório: {diretorio_arquivos}")
        return

    contador_global = 1  # Contador global para manter a numeração entre arquivos
    for arquivo in arquivos_csv:
        try:
            print(f"Processando arquivo: {arquivo}")
            contador_global = dividir_csv_por_tamanho(
                arquivo, pasta_saida, tamanho_max_mb, encoding, contador_global
            )
        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivo}: {e}")

# Exemplo de uso
diretorio_arquivos = r"C:\Users\user\Downloads\challenge-webmedia-e-globo-2023\data-tc5\files\treino"
pasta_saida = r"C:\Users\user\Downloads\challenge-webmedia-e-globo-2023\data-tc5\arquivos_divididos_treino"

processar_pasta(diretorio_arquivos, pasta_saida, tamanho_max_mb=25)
