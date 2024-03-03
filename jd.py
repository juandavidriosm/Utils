# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:49:08 2022

@author: Juan David
Módulo pasado por profesor Guillermo Villarino con múltiples cambios propios
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import winsorize
import scipy.stats as stats
from itertools import combinations
from collections import Counter
from scipy.stats import chi2_contingency
import math
import patsy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import creditpy


from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from num2words import num2words


def grafica_relacion_cat_obj(df, columna,target, order = False,titulo = False,to_words=False,bool_fix=False, dame_chi_cuadrado = False,size=(30,10)): 
    if dame_chi_cuadrado:     
        ct= pd.crosstab(df[target], df[columna]).to_numpy()
        chi2, p_val , dof, ex = stats.chi2_contingency(ct)
        if titulo == False:
              titulo = f"La relación entre {columna} y {target} tiene un p-valor en su chi^2 de: {p_val} "
    
    df = df[[columna,target]]
    if to_words:
        df[columna] = df[columna].apply(lambda x: num2words(x, lang="es") if not pd.isna(x) else "cero")
    if bool_fix:
           df[columna] = df[columna].apply(lambda x: "Verdadero" if x else "Falso")

    df_barras = df.groupby([columna], as_index = False).agg(["mean","count"])
    df_barras.columns = list(map("".join, df_barras.columns.values))
    
    if order:
        df_barras = df_barras.reset_index(inplace = False).sort_values(by = target+"mean")
    else:
        df_barras.reset_index(inplace = True)

    fig, ax = plt.subplots(figsize = size)
    bars = ax.bar(df_barras[columna], df_barras[target+"mean"])

    for bar,count in zip(bars.patches, df_barras[target+"count"]):
        height = bar.get_height()
        ax.annotate(f'{count}',fontsize=15, xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  
        textcoords='offset points',
        ha='center', va='bottom')
    labels = [str(x)+" " for x in df_barras[columna] ]
    ax.set_xticklabels(labels, rotation=90, ha='right',fontsize=20)
    if titulo:
        plt.title(titulo)

    if dame_chi_cuadrado:
            
            ct= pd.crosstab(df[target], df[columna]).to_numpy()
            chi2, p_val , dof, ex = stats.chi2_contingency(ct)
            ax.annotate(f'El p-valor de la relación con el target es de: {p_val}', xy = (15, 5))


    
    plt.show()


def grafica_relacion_MissingsTiempo(df,columna_temporal,titulo = "",return_dfConColMissing = False):
    
    df_analiza_Falta = df[df["target"]!=2].copy()
    col_name = str(columna_temporal) +"_missing"
    df_analiza_Falta[col_name] = df[columna_temporal].apply(lambda x: 1 if pd.isna(x) else 0)
    if return_dfConColMissing:
            df_analiza_Falta = df_analiza_Falta[[col_name,"created_month"]]
            to_return = df_analiza_Falta.copy()
            df_analiza_Falta = df_analiza_Falta.groupby('created_month').mean()
            df_analiza_Falta.plot(figsize=(30,8))
            plt.title(titulo)
            return to_return
    else:
            df_analiza_Falta = df_analiza_Falta[[col_name,"created_month"]]
            df_analiza_Falta = df_analiza_Falta.groupby('created_month').mean()
            df_analiza_Falta.plot(figsize=(30,8))
            plt.title(titulo)
""" DEPURACION Y VISUALIZACION DE DATOS """
#Funciones mías, falta trabaja con Theil
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
def get_cat_corr_mat(df, tipo="cramer"):
        df = df.select_dtypes(exclude = "object")
        columns = df.columns
        dm = pd.DataFrame(index=columns, columns=columns)
        if tipo == "cramer":
            for var1, var2 in combinations(columns, 2):
                asociacion = cramers_v(df[var1],df[var2])
                dm.loc[var1, var2] = asociacion
                dm.loc[var2, var1] = asociacion
            dm.fillna(1, inplace=True)
            return dm
        if tipo == "theils_u":
            for var1, var2 in combinations(columns, 2):
                coef_entrop1 = theils_u(df[var1],df[var2])
                coef_entrop2 = theils_u(df[var2],df[var1])
                dm.loc[var1, var2] = round(coef_entrop1,4)
                dm.loc[var2, var1] = round(coef_entrop2,4)
            dm.fillna(1, inplace=True)
            return dm


## Función para histograma con boxplot 
def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(20,12), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='red',linestyle='--')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data.dropna()),color='blue',linestyle='-')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    #if title: ax_box2.set(title=title, xlabel="")
    ax_box2.set(title=data.name, xlabel="")
    #leyenda
    plt.legend(handles=[
    plt.Line2D([], [], color='red', linestyle='--', linewidth=2, label='Mean'),
    plt.Line2D([], [], color='blue', linestyle='-', linewidth=2, label='Median')
    ], labels=['Mean (-)', 'Median (--)'])

    # Mostrar gráfico
    plt.show()
    
## Función para gráfico de barras de variables categóricas
def cat_plot(col):
     if col.dtypes == 'category':
        counts = col.value_counts(dropna = False)
        sorted_counts = counts.index.tolist()
        sorted_counts.sort()
        counts_index = [str(x) for x in sorted_counts]#.sort()

        #check if digit or str to make bar or barh
        es_digito = col.value_counts(dropna = True)
        es_digito = [str(int(x)).isdigit() if np.issubdtype(type(x), np.number) else False for x in es_digito.index]

        if all(es_digito):
            plt.bar(counts_index, counts.values)
            plt.title(col.name)
        else:
            plt.barh(counts_index, counts.values)
            plt.title(col.name)
        plt.show()
    


## Función general plot para aplicar al archivo por columnas
def plot(col):
     if col.dtypes == "object":
        print(f"Change column: {col.name} to another data type")
     elif col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_plot(col)

## Función manual de winsor con clip+quantile 
def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

## Función para gestionar outliers
def gestiona_outliers(col,clas = 'check'):
    
     print(col.name)
     # Condición de asimetría y aplicación de criterio 1 según el caso
     if abs(col.skew()) < 1:
        criterio1 = abs((col-col.mean())/col.std())>3
     else:
        criterio1 = abs((col-col.median())/stats.median_abs_deviation(col))>8 ## Cambio de MAD a stats.median_abs_deviation(col)
     
     # Calcular primer cuartil     
     q1 = col.quantile(0.25)  
     # Calcular tercer cuartil  
     q3 = col.quantile(0.75)
     # Calculo de IQR
     IQR=q3-q1
     # Calcular criterio 2 (general para cualquier asimetría)
     criterio2 = (col<(q1 - 3*IQR))|(col>(q3 + 3*IQR))
     lower = col[criterio1&criterio2&(col<q1)].count()/col.dropna().count()
     upper = col[criterio1&criterio2&(col>q3)].count()/col.dropna().count()
     # Salida según el tipo deseado
     if clas == 'check':
            return(lower*100,upper*100,(lower+upper)*100)
     elif clas == 'winsor':
            return(winsorize_with_pandas(col,(lower,upper)))
     elif clas == 'miss':
            print('\n MissingAntes: ' + str(col.isna().sum()))
            col.loc[criterio1&criterio2] = np.nan
            print('MissingDespues: ' + str(col.isna().sum()) +'\n')
            return(col)



# Función para calcular VCramer (dos nominales de entrada!)
def cramers_v(var1, varObj):
    if not var1.dtypes.name == 'category':
        #bins = min(5,var1.value_counts().count())
        var1 = pd.cut(var1, bins = 5)
    if not varObj.dtypes.name == 'category': #np.issubdtype(varObj, np.number):
        #bins = min(5,varObj.value_counts().count())
        varObj = pd.cut(varObj, bins = 5)
        
    data = pd.crosstab(var1, varObj).values
    vCramer = stats.contingency.association(data, method = 'cramer')
    return vCramer


# Ejemplo uso univariante
#cramers_v(vinosCompra['Etiqueta'],vinosCompra['Beneficio'])

# Aplicar la función al input completo contra la objetivo
#tablaCramer = pd.DataFrame(imputCompra.apply(lambda x: cramers_v(x,varObjCont)),columns=['VCramer'])


## Función mejor tranformación ##
# Busca la transformación de variables input de intervalo que maximiza la VCramer o 
# la correlación tipo Pearson con la objetivo
def mejorTransf (vv,target, name=False, tipo = 'cramer', graf=False, escalar = True, evita_ceros = False):
    if evita_ceros:
        if 0 in vv.values:
            vv += 0.0001
    
    
    if escalar:
        # Escalado de datos (evitar fallos de tamaño de float64 al hacer exp de número grande..cosas de python)
        vv = pd.Series(scale(vv), name=vv.name)
        # Traslación a valores positivos de la variable (sino falla log y las raíces!)
        vv = vv + abs(min(vv))+0.0001
    
    
      
    # Definimos y calculamos las tranformacione típicas  
    transf = pd.DataFrame({vv.name + '-ident': vv, vv.name + '-log': np.log(vv), vv.name + '-exp': np.exp(vv), 
                         vv.name + '-sqrt': np.square(vv), vv.name + '-cuarta': vv**4, vv.name + '-raiz4': vv**(1/4)})
      
    # Distinguimos caso cramer o caso correlación
    if tipo == 'cramer':
      # Aplicar la función cramers_v a cada trasnformación frente a la respuesta
      tablaCramer = pd.DataFrame(transf.apply(lambda x: cramers_v(x,target)),columns=['VCramer'])
      
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCramer,x=tablaCramer.VCramer,title='Relaciones frente a ' + target.name +" de transformaciones de la variable "+ vv.name).update_yaxes(categoryorder="total ascending").show()
      # Identificar mejor transfromación
      best = tablaCramer.query('VCramer == VCramer.max()').index
      ser = transf[best[0]].squeeze()
    
    if tipo == 'cor':
      # Aplicar coeficiente de correlacion a cada trasnformación frente a la respuesta
      tablaCorr = pd.DataFrame(transf.apply(lambda x: np.corrcoef(x,target)[0,1]),columns=['Corr'])
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCorr,x=tablaCorr.Corr,title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # identificar mejor transfromación
      best = tablaCorr.query('Corr.abs() == Corr.abs().max()').index
      ser = transf[best[0]].squeeze()
  
    # Aquí distingue si se devuelve la variable transfromada o solamente el nombre de la transfromacion
    if name:
      return(ser.name)
    else:
      return(ser)

# Ejemplo de uso univariante
#tr = mejorTransf(vinosCompra.Azucar,varObjCont, tipo='cor')


"""    REGRESIONES Y MODELIZACIÓN     """


# Función para generar la fórmula por larga que sea
def ols_formula(df, dependent_var, *excluded_cols):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)

def ols_formula_list(df, dependent_var, lista, keep = False):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    if keep:
        for col in df_columns:
            if col not in lista:
                df_columns.remove(col)
    else:  
        for col in df_columns:
            if col in lista:
                df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)

# Aplicamos a fórmula de modelo completo
#form=ols_formula(data_train,'Beneficio')

# Función para evaluación de modelos Logísticos en training-test (input: fórmula y dataset al natural)
def tr_tst_eval_lin(formula,data):
  # Generamos las matrices de diseño según la fórmula de modelo completo
  y, X = patsy.dmatrices(formula, data, return_type='dataframe')
  
  # Creamos 4 objetos: predictores para tr y tst y variable objetivo para tr y tst. 
  X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=1234)
  
  # Definición de modelo
  modelo =  LinearRegression()
  
  # Ajuste de modelo
  modelo = modelo.fit(X_tr,y_tr)
  
  # Accuracy del modelo en training
  r2 = modelo.score(X_tr,y_tr)
  print('Coeficiente de determinación TRAINING: ',r2, '\n')
  
  # Predicciones en test
  y_pred = modelo.predict(X_tst)

  # Cálculo de performance
  print("Mean squared error TEST: %.2f" % np.sqrt(mean_squared_error(y_tst, y_pred)))
  print("Coeficiente de determinación TEST: %.2f" % r2_score(y_tst, y_pred))


# Función para evaluación de modelos Logísticos en training-test (input: fórmula y dataset al natural)
def tr_tst_eval_log(formula,data):
  # Generamos las matrices de diseño según la fórmula de modelo completo
  y, X = patsy.dmatrices(formula, data, return_type='dataframe')
  
  # Creamos 4 objetos: predictores para tr y tst y variable objetivo para tr y tst. 
  X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=1234)
  
  # Definición de modelo
  modelo = LogisticRegression(solver='lbfgs', max_iter=2000,penalty='none')
  
  # Arreglar y para que le guste a sklearn...numeric
  y_tr_ = y_tr.iloc[:,0].ravel()
  
  # Ajuste de modelo
  modelLog = modelo.fit(X_tr,y_tr_)
  
  # Accuracy del modelo en training
  acc = modelLog.score(X_tr,y_tr_)
  print('Accuracy en training: ',acc, '\n')
  
  # Predicciones en test
  y_pred = modelLog.predict(X_tst)
  
  # Matriz de confusion de clasificación 
  print('Matriz de confusión y métricas derivadas: \n',metrics.confusion_matrix(y_tst,y_pred))
  
  # Reporte de clasificación 
  print(metrics.classification_report(y_tst,y_pred))
  
  # Extraemos el Area bajo la curva ROC
  print('Area bajo la curva ROC training: \n', metrics.roc_auc_score(y_tr, modelLog.predict_proba(X_tr)[:, 1]))
  print('Area bajo la curva ROC test: \n', metrics.roc_auc_score(y_tst, modelLog.predict_proba(X_tst)[:, 1]))


# Función para comparación por validación cruzada
def cross_val_lin(formula, data, seed=12345):
      # Generamos las matrices de diseño según la fórmula de modelo completo
      y, X = patsy.dmatrices(formula, data, return_type='dataframe')
      
      model = LinearRegression()
      
      # Establecemos esquema de validación fijando random_state (reproducibilidad)
      cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)
  
      # Obtenemos los resultados de R2 para cada partición tr-tst
      scores = cross_val_score(model, X, y, cv=cv)
  
      # Sesgo y varianza
      print('Modelo: ' + formula)
      print('Coeficiente de determinación R2: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
      
      #sns.violinplot(y=scores,palette='viridis')
      
      return(scores)

# Función para comparación por validación cruzada
def cross_val_log(formula, data, seed=12345):
    # Generamos las matrices de diseño según la fórmula de modelo completo
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    y = y.iloc[:,0].ravel()
      
    model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty=None)
  
    # Establecemos esquema de validación fijando random_state (reproducibilidad)
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)
     
    # metrics.get_scorer_names() --> Posibilidades de distintas métricas! 
      
    # Obtenemos los resultados de R2 para cada partición tr-tst
    scores_AUC = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
    scores_Accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    scores_F1 = cross_val_score(model, X, y, scoring='f1', cv=cv)
  
    # Sesgo y varianza
    print('Modelo: ' + formula)
    print('AUC: %.3f (%.3f)' % (np.mean(scores_AUC), np.std(scores_AUC)))
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores_Accuracy), np.std(scores_Accuracy)))
    print('F1: %.3f (%.3f)' % (np.mean(scores_F1), np.std(scores_F1)))
      
    sns.violinplot(y=scores_Accuracy,palette='viridis').set_title('Accuracy in CV')
      
    #return(scores)


def cross_val_model(model, X, y, seed=123):

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)      

    scores_AUC = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
    scores_Accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    scores_F1 = cross_val_score(model, X, y, scoring='f1', cv=cv)
  
    # Sesgo y varianza
    print('AUC: %.3f (%.3f)' % (np.mean(scores_AUC), np.std(scores_AUC)))
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores_Accuracy), np.std(scores_Accuracy)))
    print('F1: %.3f (%.3f)' % (np.mean(scores_F1), np.std(scores_F1)))
      
    sns.violinplot(y=scores_Accuracy,palette='viridis').set_title('Accuracy in CV')

    #return scores
  

  
# Función para pintar la curva ROC
def roc_grafico(test,pred): 
    fpr, tpr, thresholds = metrics.roc_curve(test,pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
 
# Función pto de corte por Youden
def cutoff_youden(test,pred):
    fpr, tpr, thresholds = metrics.roc_curve(test,pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

# Función para comparación por validación cruzada válido para selección
# de variables por sfs o Lasso o clualquier tipo de dataset reducido en número de variables.
def cross_val_selectVar(sfs, data, y, log=False, seed=12345):
        # Por defecto, digamos que la entrada es una matriz explícita de variables seleccionadas
        X = sfs
	
	# Si realmente es un objeto sfs, obtenemos matriz explícita con transform
        if not isinstance(sfs,pd.DataFrame):
            X = sfs.transform(data)
        # Para logística arreglar variable objetivo para sklearn, definit logistic y métrica    
        if log: 
            y.iloc[:,0].ravel()
            model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='none')
            scoring='roc_auc'
        # En caso contrario es regresión lineal y lo especificamos
        else: 
            model = LinearRegression()
            scoring='r2'
        
        # Establecemos esquema de validación fijando random_state (reproducibilidad)
        cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=seed)

        # Obtenemos los resultados de R2 para cada partición tr-tst
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)

        # Sesgo y varianza
        print('Métrica ' + scoring + ': %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

       # sns.violinplot(y=scores,palette='viridis')

        return(scores)
    

"""       SERIES TEMPORALES     """


# Función para pasar el test de estacionariedad de Dickey Fuller
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# Función para pintar el gráfico estacional
def seasonal_plot(df, season='year', index='month', column=None):
    """Makes a seasonal plot of one column of the input dataframe. Considers the first columns by default.
    
    Arguments:
    
    - df (Pandas DataFrame): DataFrame indexed by Datetime (see `parse_dates` parameter when reading a CSV);
    - season (string): the season that you want to considering when doing the plot, e.g., year, month, etc.;
    - index (string): corresponds to the X axis of the plot. You should choose based on the index period that you're using;
    - column (string, optional): the DataFrame column to consider. Picks the first one by default.
    """
    if column == None:
        column = df.columns[0]
        
    piv_index = getattr(df.index, index)
    piv_season = getattr(df.index, season)
    
    piv = pd.pivot_table(df, index=piv_index, columns=piv_season, values=[column])
    piv.plot(figsize=(12,8))
    

# Función para evaluar modelo de suavizado o arima manual (No válido para auto_arima)
def eval_model(model,tr,tst,name='Model',lags=12):
    lb = np.mean(sm.stats.acorr_ljungbox(model.resid, lags=lags, return_df=True).lb_pvalue)
    pred = model.forecast(steps=len(tst))
    plt.tight_layout()
    fig1, ax = plt.subplots()
    ax.plot(tr, label='training')
    ax.plot(tst, label='test')
    ax.plot(pred, label='prediction')
    plt.legend(loc='upper left')
    tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(round(mean_absolute_percentage_error(tst, pred)*100,2)) + "%"
    plt.title(tit)
    plt.ylabel('Pasajeros')
    plt.xlabel('Date')
    plt.show()
    figsize=(20, 6)
    print(lb)
 
# Función para evaluar modelo auto_arima
def eval_model_Aarima(model,tr,tst,name='Model',lags=12):
      lb = np.mean(sm.stats.acorr_ljungbox(model.resid(), lags=lags, return_df=True).lb_pvalue)
      pred = model.predict(n_periods=len(tst))
      fig1, ax = plt.subplots()
      ax.plot(tr, label='training')
      ax.plot(tst, label='test')
      ax.plot(pred, label='prediction')
      plt.legend(loc='upper left')
      tit = name + ":  LjungBox p-value --> " + str(lb) + "\n MAPE: " + str(round(mean_absolute_percentage_error(tst, pred)*100,2)) + "%"
      plt.title(tit)
      plt.ylabel('Serie')
      plt.xlabel('Date')
      plt.show()
      model.plot_diagnostics(figsize=(14,10))
      plt.show()  


# Función para evaluar residuos a través de contrastes de hipótesis
def residcheck(residuals, lags):
    """
    Function to check if the residuals are white noise. Ideally the residuals should be uncorrelated, zero mean, 
    constant variance and normally distributed. First two are must, while last two are good to have. 
    If the first two are not met, we have not fully captured the information from the data for prediction. 
    Consider different model and/or add exogenous variable. 
        
    If Ljung Box test shows p> 0.05, the residuals as a group are white noise. Some lags might still be significant. 
        
    Lags should be min(2*seasonal_period, T/5)
        
    plots from: https://tomaugspurger.github.io/modern-7-timeseries.html
        
    """
    resid_mean = np.mean(residuals)
    lj_p_val = np.mean(sm.stats.acorr_ljungbox(x=residuals, lags=lags).lb_pvalue)
    norm_p_val =  stats.jarque_bera(residuals)[1]
    adfuller_p = adfuller(residuals)[1]
        
      
    fig = plt.figure(figsize=(10,8))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2);
    acf_ax = plt.subplot2grid(layout, (1, 0));
    kde_ax = plt.subplot2grid(layout, (1, 1));
    
    residuals.plot(ax=ts_ax)
    plot_acf(residuals, lags=lags, ax=acf_ax);
    sns.kdeplot(residuals);
    #[ax.set_xlim(1.5) for ax in [acf_ax, kde_ax]]
    sns.despine()
    plt.tight_layout();
    plt.show()
    print("** Mean of the residuals: ", np.around(resid_mean,2))
        
    print("\n** Ljung Box Test, p-value:", np.around(lj_p_val,3), 
        "(>0.05, Uncorrelated)" if (lj_p_val > 0.05) else "(<0.05, Correlated)")
        
    print("\n** Jarque Bera Normality Test, p_value:", np.around(norm_p_val,3),
        "(>0.05, Normal)" if (norm_p_val>0.05) else "(<0.05, Not-normal)")
        
    print("\n** AD Fuller, p_value:", np.around(adfuller_p,3), 
        "(>0.05, Non-stationary)" if (adfuller_p > 0.05) else "(<0.05, Stationary)")
    
    return ts_ax, acf_ax, kde_ax   


"""    REDUCCION DE DIMENSIONES  """

# Dibujar biplot 
def biplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley) #, c = cities.index.tolist())
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

# Ejemplo de uso. Siempre para 2 componentes!
#biplot(scores[:,0:2],np.transpose(pca.components_[0:2, :]),cities.columns)
#plt.show()


"""    CLUSTERING    """

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from time import time

# Extraer dendograma y pintarlo
def plot_dendogram(model, **kwargs):
  '''
  Esta función extrae la información de un modelo AgglomerativeClustering
  y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
  '''
      
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1  # leaf node
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count
  
  linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)
  
  # Plot
  dendrogram(linkage_matrix, **kwargs)
  plt.show()
  
  
# Cremos función scree_plot_kmeans para buscar el número de clusters óptimo 
# con 3 métricas usuales. Wss, silueta y % de varianza explicada
# Input: data = dataset en bruto (se escala dentro de la propia función)
#        n_max = número máximo de grupos a evaluar
# ==============================================================================

def scree_plot_kmeans(data,n_max):
  range_n_clusters = range(2, n_max)
  X_scaled = scale(data)
  inertias = []
  silhouette = []
  var_perc = []
  
  for n_clusters in range_n_clusters:
      modelo_kmeans = KMeans(
                          n_clusters   = n_clusters, 
                          n_init       = 20, 
                          random_state = 123
                      )
      modelo_kmeans.fit(X_scaled)
      cluster_labels = modelo_kmeans.fit_predict(X_scaled)
      inertias.append(modelo_kmeans.inertia_)
      silhouette.append(silhouette_score(X_scaled, cluster_labels))
      tss = sum(pdist(X_scaled)**2)/X_scaled.shape[0]
      bss = tss - modelo_kmeans.inertia_
      var_perc.append(bss/tss*100)
      
  fig, ax = plt.subplots(1, 3, figsize=(16, 6))
  ax[0].plot(range_n_clusters, inertias, marker='o')
  ax[0].set_title("Scree plot Varianza intra")
  ax[0].set_xlabel('Número clusters')
  ax[0].set_ylabel('Intra-cluster (inertia)')
  
  ax[1].plot(range_n_clusters, silhouette, marker='o')
  ax[1].set_title("Scree plot silhouette")
  ax[1].set_xlabel('Número clusters')
  ax[1].set_ylabel('Media índices silhouette');
  
  ax[2].plot(range_n_clusters, var_perc, marker='o')
  ax[2].set_title("Scree plot % Varianza")
  ax[2].set_xlabel('Número clusters')
  ax[2].set_ylabel('% de varianza explicada')
  plt.tight_layout()
