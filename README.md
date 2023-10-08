## DETECÇÃO AUTOMÁTICA DE EXOPLANETAS UTILIZANDO ALGORITMOS DE APRENDIZADO DE MÁQUINA

**Bruno Henrique Dourado Macedo(1)*, Joylan Nunes Maciel(2), Willian Zalewski(3)**

**(1)** Voluntário (IC/Af), Engenharia Física, Instituto Latino-Americano De Ciências Da Vida E Da Natureza (ILACVN), UNILA.

**(2)** Coorientador(a), Instituto Latino-Americano de Tecnologia, Infraestrutura e Território (ILATIT), Universidade Federal da Integração Latino Americana, UNILA.

**(3)** Orientador(a), Instituto Latino-Americano de Tecnologia, Infraestrutura e Território (ILATIT) Instituto Latino-Americano De Ciências Da Vida E Da Natureza, UNILA.

*E-mail de contato: bhd.macedo.2017@aluno.unila.edu.br*

### 1. RESUMO

Nas últimas décadas ocorreu um avanço tecnológico alinhado à redução de custos na instrumentação astronômica, gerando um grande aumento na capacidade de coleta e armazenamento de dados pelos cientistas. Diversas tecnologias possibilitaram a varredura sistemática de galáxias, e impulsionaram novas descobertas sobre diversos fenômenos existentes no universo. Em especial, as missões espaciais como a CoRoT , NuSTAR, NEOWISE, Gaia, Hubble, Kepler, TESS e o mais recente o Telescópio Espacial James Webb contribuíram significativamente para esses avanços.

Os principais tipos de dados coletados nessas missões espaciais são séries temporais denominadas curva de luz. Este tipo de dado pode ser entendido como um conjunto ordenado de observações, registradas cronologicamente, da intensidade luminosa de corpos celestes. Em especial, as curvas de luz podem ser aplicadas na identificação de exoplanetas por meio do método de trânsito planetário. Utilizando esta técnica, foram descobertos 76% dos exoplanetas encontrados a partir dos dados do Kepler.

Contudo, o contínuo armazenamento de dados, em especial na forma de curvas de luz, tem promovido o rápido crescimento de uma enorme quantidade de dados, e como consequência, as técnicas de análise tradicionais tornaram-se inviáveis para a exploração eficiente desses dados. Como exemplo, o projeto espacial Kepler da NASA, totalizou cerca de 678 GB de dados coletados ao final do projeto.

Avaliando o cenário mencionado, o objetivo deste trabalho foi contribuir para o processo de tomada de decisão de astrônomos, por meio da detecção automática de exoplanetas, utilizando algoritmos de aprendizado de máquina. Com este intuito, neste trabalho, foi elaborada uma avaliação experimental por meio da plataforma Google Colab, usando bibliotecas para a linguagem de programação Python (Scikit-Learn, Sktime, lightkurve, pandas, numpy).

Inicialmente foi estruturado um banco de dados com curvas de luz provenientes do catálogo online NASA Exoplanet Archive. Neste estudo experimental os dados foram pré-processados para redução de dimensionalidade utilizando duas estratégias, local e global. Os algoritmos de aprendizagem de máquina selecionados para avaliação foram: Decision Trees (DT), Support Vector Machines (SVM), Random Forest (RF), Naive Bayes (NB), Nearest Neighbors (NN) e Neural Networks (MLP). Para cada um desses algoritmos foi aplicada a abordagem Grid Search, a qual busca os melhores parâmetros a serem utilizados para a indução dos modelos.

A avaliação dos modelos induzidos foi realizada por meio da estratégia Cross-Validation (Validação Cruzada) com 10 partições. Essa estratégia de avaliação foi experimentalmente repetida 30 vezes. Em cada repetição dados diferentes para cada partição foram selecionados, minimizando assim um possível viés sobre os dados. Os melhores resultados para acurácia foram obtidos utilizando o algoritmo Neural Networks, com 77,65% para os dados globais e 70,32% para os dados locais. Em trabalhos futuros, pretendemos utilizar uma base de dados com uma menor divergência na quantidade de itens e na elaboração de novos algoritmos que se baseiam em redes neurais e deep learning.

## Tabela 1. Resultado da média e o desvio padrão.

### Global

| Modelo | Acurácia | Precisão | Recall | F1    | Média (%) | Desvio (%) | Média (%) | Desvio (%) |
|--------|---------|----------|--------|-------|-----------|------------|-----------|------------|
| SVM    | 74,06   | 3,73     | 78,55  | 2,21  | 76,65     | 7,85       | 77,39     | 4,18       |
| RF     | 72,20   | 3,03     | 75,59  | 1,85  | 77,39     | 6,82       | 76,48     | 3,54       |
| DT     | 68,13   | 1,87     | 72,99  | 1,91  | 72,03     | 5,63       | 72,54     | 2,60       |
| NB     | 70,54   | 7,96     | 79,93  | 3,16  | 66,47     | 17,55      | 71,30     | 10,97      |
| NN     | 71,01   | 2,33     | 79,93  | 2,72  | 67,69     | 5,21       | 73,15     | 2,87       |
| MLP    | 77,65   | 4,17     | 82,42  | 4,12  | 78,26     | 10,62      | 80,39     | 4,83       |

### Local

| Modelo | Acurácia | Precisão | Recall | F1    | Média (%) | Desvio (%) | Média (%) | Desvio (%) |
|--------|---------|----------|--------|-------|-----------|------------|-----------|------------|
| SVM    | 68,85   | 2,21     | 74,57  | 1,46  | 71,59     | 5,15       | 72,52     | 2,83       |
| RF     | 66,01   | 1,22     | 70,58  | 0,73  | 71,98     | 1,79       | 71,26     | 1,27       |
| DT     | 63,38   | 1,10     | 68,80  | 0,64  | 68,39     | 2,96       | 68,68     | 1,62       |
| NB     | 68,38   | 1,95     | 80,31  | 1,58  | 60,97     | 3,21       | 69,27     | 2,41       |
| NN     | 66,64   | 1,45     | 69,05  | 1,61  | 78,23     | 2,61       | 73,31     | 1,14       |
| MLP    | 70,32   | 1,46     | 74,90  | 1,13  | 74,14     | 3,84       | 74,17     | 1,76       |

Fonte: Autoria própria.

## 3. AGRADECIMENTOS

Agradeço a UNILA por abrir as portas da universidade, ao professor Willian Zalewski pela orientação
neste trabalho, e aos professores da engenharia física e aos meus ex-colegas Patricia e Victor pela
ajuda e a minha companheira Renata Benedet pelo apoio.

Link para download do anais do SIEPE 2022: https://portal.unila.edu.br/eventos/siepe-2022/arquivos/2022_ANAIS_SIEPE_c.pdf
