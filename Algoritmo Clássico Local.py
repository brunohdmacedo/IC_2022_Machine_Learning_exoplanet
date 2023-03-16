#TESTE COM FOR ENTRE 0 E 31

book_name = 'classicos_local'
pos_label = 1
nome = 0
for i in range(0,31):
      nome = i
      from sklearn.model_selection import StratifiedKFold
      cv = StratifiedKFold(10, random_state=1, shuffle=True)

      data_path = '/content/drive/MyDrive/Iniciação Científica/2021-2022/IC_Exoplanetas_2022_Experimento/Base de Dados/shallue_local_curves_.csv'
      path_result = '/content/drive/MyDrive/Iniciação Científica/2021-2022/IC_Exoplanetas_2022_Experimento/Base de Dados/Resultados teste/Local/Clássico/' + book_name + '_' + str(nome) + '.xlsx'

      import numpy as np
      import pandas as pd
      from openpyxl import load_workbook

      from sklearn import tree
      from sklearn.svm import SVC
      from sklearn.naive_bayes import GaussianNB
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.neural_network import MLPClassifier

      from sklearn.preprocessing import LabelBinarizer
      from sklearn.model_selection import RandomizedSearchCV

      from sklearn.metrics import make_scorer
      from sklearn.metrics import accuracy_score,precision_score, average_precision_score
      from sklearn.metrics import f1_score,recall_score,roc_auc_score,balanced_accuracy_score

      data = pd.read_csv(data_path, sep = ",") 

      #definição input e label no formato tabular exigido pelo scikit-learn
      data_input = data.copy()
      label = data_input.pop(data_input.columns[len(data_input.columns)-1])

      X = data_input.values
      y = label.values
      #normalização
      norm_data = data_input.copy()
      norm_data = norm_data.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
      X_norm = norm_data.values #tamanho limitado para testes rápidos

      #label binário
      lb = LabelBinarizer()
      y = lb.fit_transform(label)
      y = y.reshape(-1) #tamanho limitado para testes rápidos

      #definição dos modelos e parametros
      model_params = {
          'svm': {
              'model': SVC(gamma='auto'),
              'params' : {
                  'C': [1,3,5,10,15],
                  'kernel': ['rbf','linear'],
                  'tol' : [1e-3,1e-4],
                  'random_state': [1]
              }  
          },
          'random_forest' : {
              'model': RandomForestClassifier(),
              'params' : {
                  'n_estimators': [1,3,5,8,10],
                  'max_features': ['sqrt','log2'],
                  'random_state': [1]
              }
          },
          'decision_tree' : {
              'model': tree.DecisionTreeClassifier(),
              'params': {
                  'max_features': ['sqrt', 'log2'],
                  'min_samples_split': [2,4,6,8,10],
                  'random_state': [1]
              }
          },
          'naive_bayes' : {
              'model': GaussianNB(),
              'params': {
                  'var_smoothing': [1e-09,1e-12,1e-15]
              }
          },
          'k_neighbors' : {
              'model': KNeighborsClassifier(),
              'params': {
                  'n_neighbors': [1],
                  'algorithm' : ['ball_tree','kd_tree','brute']
              }
          },
          'MLPClassifier': {
              'model': MLPClassifier(),
              'params':{
                  'random_state': [1],
                  'tol' : [1e-3,1e-4,200],
                  'solver': ['sgd', 'adam']
              }
          }
      }

      #definição das métricas e parametros
      scoring = {'acc': 'accuracy',
                'prec': make_scorer(precision_score,pos_label=pos_label),
                'avg_prec': make_scorer(average_precision_score,pos_label=pos_label),
                'recall': make_scorer(recall_score,pos_label=pos_label),
                'f1': make_scorer(f1_score,pos_label=pos_label),
                'bal_acc': 'balanced_accuracy'
                  }

      #execução dos modelos com randomizedsearchcv
      scores = []
      for model_name, mp in model_params.items():
          clf = RandomizedSearchCV(mp['model'], mp['params'], cv=cv, scoring=scoring, return_train_score=True, refit=False, n_iter=3)
          clf.fit(X_norm, y)

          model_dic = {'model': model_name}
          score = clf.cv_results_
          metrics = {**model_dic, **score}
          mtrc = pd.DataFrame(metrics)
          scores.append(mtrc)

      #resultados em dataframe
      lista = [scores[0],scores[1],scores[2],scores[3],scores[4],scores[5]] 
      resultados_completos = pd.concat(lista, ignore_index=True)

      #dataframe com seleção de médias e desvio padrões
      resultados= pd.DataFrame()
      resultados[['model','mean_test_acc','std_test_acc','mean_test_prec','std_test_prec',
                  'mean_test_avg_prec','std_test_avg_prec','mean_test_recall','std_test_recall',
                  'mean_test_f1','std_test_f1','mean_test_bal_acc',
                  'std_test_bal_acc']] = resultados_completos[['model','mean_test_acc',
                                              'std_test_acc','mean_test_prec','std_test_prec','mean_test_avg_prec',
                                              'std_test_avg_prec','mean_test_recall','std_test_recall','mean_test_f1',
                                              'std_test_f1','mean_test_bal_acc','std_test_bal_acc']]

      #salva dataframes no excel
      resultados.to_excel(path_result, sheet_name='local')  

      book = load_workbook(path_result)
      writer = pd.ExcelWriter(path_result, engine='openpyxl')
      writer.book = book

      resultados_completos.to_excel(writer, sheet_name='_completo')
      writer.save()
      writer.close()

print('OK!!!!!')