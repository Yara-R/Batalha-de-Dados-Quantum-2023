import numpy as np
import pandas as pd

from docplex.mp.model_reader import ModelReader
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

def prepara_amostra_problema(seed: int) -> tuple:
    """Função que prepara uma amostra para
    definir um problema de otimização para ser
    resolvido usando computação quântica.

    Args:
        seed (int): Seed.

    Returns:
        tuple: Uma tupla onde o primeiro elemento
        é o DataFrame da amostra e o segundo é a
        matriz de covariância multiplicada pela
        alocação. 
    """

    dados = pd.read_csv(filepath_or_buffer='dados_batalha_quantum.csv',
                        index_col=0)
    covariancia = pd.read_csv(filepath_or_buffer="matriz_covariancia_produtos_quantum.csv",
                              index_col=0)

    np.random.seed(seed)

    cols = ['produto',
            'classe',
            'ret_med',
            'aval',
            'aloc']
    dict_alocacao = {"classe_1": 0.3,
                     "classe_2": 0.3,
                     "classe_3": 0.4}
    
    sample = dados[dados.classe.isin(dict_alocacao.keys())].groupby("classe").sample(2, random_state=seed)
    sample['aval'] = (10*sample.aval/sample.aval.sum()).round(1).astype(int)
    sample = sample[cols].copy()
    
    aloc_matriz = np.outer(sample.aloc.values, sample.aloc.values)
    cov_sample = covariancia.loc[sample.produto.values][sample.produto.values].values*aloc_matriz
    
    return sample, cov_sample

def carrega_problema_qubo(seed: int) -> QuadraticProgram:
    """Carrega o problema qubo relacionado
    a seed de input.

    Args:
        seed (int): Seed.

    Returns:
        QuadraticProgram: Problema qubo
    """
    
    lp_file = ModelReader.read(f"qubo_sample{seed}.lp")
    qubo = from_docplex_mp(lp_file)

    return qubo

def gera_relatorio(seed: int,
                   variaveis_resposta: np.ndarray) -> None:
    """Exibe um relatório sobre o resultado da
    otimização.

    Args:
        seed (int): Seed.
        variaveis_resposta (np.ndarray): variáveis respostas
        obtidas na resolução do problema de otimização.
    """
    
    colunas = ["variaveis_resposta",
               "funcao_objetivo",
               "quantidade_produtos",
               "avaliacao_risco",
               "risco",
               "retorno",
               "sharpe",
               "alocacao"]
    
    sample, cov_sample = prepara_amostra_problema(seed=seed)
    mu = sample.ret_med.to_numpy()
    risk = np.sqrt(variaveis_resposta@cov_sample@variaveis_resposta)
    ret = variaveis_resposta@mu
    f_obj = risk**2 - ret
    
    data = [variaveis_resposta,
             f_obj,
             variaveis_resposta.sum(),
             sample[variaveis_resposta.astype(bool)].aval.sum(),
             risk,
             ret,
             ret/risk,
             sample[variaveis_resposta.astype(bool)].aloc.sum()]
    
    print('-'*100)
    print("Estatísticas da carteira")
    print('-'*100)
    for key, value in zip(colunas, data):
        print(f"{key}: {value}")

    print('-'*100)
    print("Produtos selecionados")
    print('-'*100)
    print(sample[variaveis_resposta.astype('bool')][['classe', 'produto', 'aval', 'aloc']])
