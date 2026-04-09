## Modelos implementados 

### 1 Modelo LSTM (`model_lstm.py`)
 O modelo LSTM proposto utiliza os quatro valores mais recentes da série temporal para prever as três semanas seguintes de casos prováveis. Sua arquitetura é composta por três camadas LSTM com 64 hidden units cada, conforme ilustrado na Figura 3. O modelo emprega como preditores variáveis derivadas da série de casos, incluindo a diferença de primeira ordem, média, desvio padrão e coeficiente angular calculados em janelas deslizantes de quatro semanas, além da semana epidemiológica. Antes do treinamento, a série foi transformada utilizando a transformação de Box-Cox, o que auxilia na estabilização da variância e na melhoria do ajuste do modelo.

### 2 Modelo de Processo Gaussiano (`model_gp.py`)
 O modelo de processo gaussiano foi treinado com os mesmos preditores utilizados no LSTM e também com a série transformada pela função de Box-Cox. Seu kernel é composto pela soma de um Matern 3/2 kernel — aplicado a todos os preditores — e um kernel periódico ajustado exclusivamente à variável semana epidemiológica (SE), permitindo capturar tanto padrões locais quanto sazonais da série. A implementação foi realizada com o pacote gpflow, e, como o modelo não prevê múltiplos passos de forma direta, ele foi configurado para estimar o número de casos três semanas à frente, sendo aplicado em formato de janela deslizante para gerar as previsões sucessivas.

### 3 Modelo ARIMA (`model_arima.py`)
O modelo ARIMA adotado é univariado e não sazonal, sendo ajustado diretamente à série de casos prováveis. Sua implementação foi realizada por meio do pacote mosqlient, que estima automaticamente os parâmetros do modelo com base no critério de informação de Akaike (AIC), buscando o melhor equilíbrio entre ajuste e complexidade. Por se tratar de um modelo estatístico clássico, o ARIMA utiliza apenas a dependência temporal da própria série, sem recorrer a variáveis preditoras adicionais.
Método de ensemble
Cada modelo acima fornece as previsões 3 semanas à frente com a mediana e o intervalo de confiança de 95%.  
Os passos para a aplicação do ensemble estão apresentados abaixo:
- Para cada modelo e ponto predito são aproximados os parâmetros de uma distribuição log-normal;  
- A distribuição resultante é uma combinação logarítmica da distribuição de cada modelo com pesos iguais. 


Para formatar os dados é necessário ter acesso a tabelas do infodengue separadas por estado em arquivos `.parquet`. Após isso: 
* `format_data.py` - gerar os arquivos para aplicação dos modelos;
* `train_models.py` - treinamento dos modelos;
* `apply_models.py` - aplicação dos modelos.

Geração das figuras por estado para os informes: `new_figures.ipynb.`
