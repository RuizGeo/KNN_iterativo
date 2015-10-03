# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:34:56 2015

@author: ruiz
"""
import numpy as np

class IterativoKNN:
    '''Metodo de classificacao k-vizinhos mais proximos iterativo '''
    def __init__(self,treinamento,novos_exemplos,peso,potencia,taxa_acerto,total_iter):
        #iniciar variaveis
        self.treinamento = treinamento #1D array numpy
        self.novos_exemplos = novos_exemplos #1D array numpy
        self.peso = peso #inteiro
        self.potencia = potencia #inteiro 
        self.taxa_acerto = taxa_acerto #real (0-1)
        self.total_iter = total_iter #inteiro
    def calcDistMinkowski(self):    
        '''Calcular a distancia entre treinamento e novos exemplos a ser classificados'''
        #criar um array colunas igual ao tamanho do treinamento e linhas igual aos novos exemplos  
        self.dist=np.arange(self.treinamento.shape[0]*self.novos_exemplos.shape[0]).reshape(self.treinamento.shape[0],self.novos_exemplos.shape[0])
        #Percorrer array novos exemplos
        for i in xrange(self.novos_exemplos.shape[0]):

            #calcular distancia         
            self.dist[i,:] = np.sum(self.peso*(abs(self.novos_exemplos[i,:]-self.treinamento)**self,potencia),1)**(1./self.potencia)
        
   
        potencias=[1]#,2,3]
    def avaliarKNN (knn):
        #percorrer linhas do array distancias
        for potencia in potencias:
            for limiar in limiares:
                #Criar a matriz de distancias e guardar em um arquivo HDF
    #            distMinkowski(datas[datas.columns[peso>limiar]].values[arrayIdSeg-1],datas[datas.columns[peso>limiar]].values,peso[peso>limiar], potencia,arrayDist)
                distMinkowski(datas[datas.columns[peso>limiar]].values[arrayIdSeg-1],datas[datas.columns[peso>limiar]].values[arrayIdSegTest-1],peso[peso>limiar], potencia,arrayDist)
    
                for k in knn:
                    
                    
                    for row in xrange(arrayDist.shape[0]):
                        #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
                        freq_classes=np.bincount(dataClassTrain[np.in1d(arrayDist[row],np.sort(arrayDist[row])[:k])])
                        k_modificado=k
                        controle=False
                        #numero de iteracoes
                        cont=0
                        while cont < 5 and controle == False:
                            
                            cont+=1
                            #print 'row: ',row
                            #print '(np.max(freq_classes)):',np.max(freq_classes)
                            #print 'SUM(freq_classes):',np.sum(freq_classes)
                            #print 'np.max(freq_classes)/np.sum(freq_classes): ',np.max(freq_classes)/float(np.sum(freq_classes))
                            #print 'k_modificado: ',k_modificado
                            
                            if np.max(freq_classes)/float(np.sum(freq_classes)) >=    0.7:
                                #print 'maior que 0.5: '
                                #Selecionar a classe para a maximo k vizinhos
                                classificacao[row+1]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]
                                controle=True                          
                           
                            else:
                                #Selecionar a classe para a maximo k vizinhos
                                #classificacao[row+1]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]
                                #soma valor ao k_modificado
                                k_modificado=k_modificado+(round(np.max(freq_classes)*0.7))
                                #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
                                freq_classes=np.bincount(dataClassTrain[np.in1d(arrayDist[row],np.sort(arrayDist[row])[:k_modificado])])
                        
                        classificacao[row+1]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]
                                
                        
            
                        
                    
                    #exatidao global
                    #create matrix confusion
                    matrix_confusion=np.histogram2d(dataClassTest,classificacao[1:],bins=(13,13))[0]
                    accuracy=inter_rater.cohens_kappa(matrix_confusion)
                    #writer_csv.writerow((str(limiar),str(k),str(potencia),str(round(accuracy.kappa,3)),str(round(accuracy.var_kappa,5))))
        
                    print 'k,limiar, potencia: ',k,limiar,potencia
                    print accuracy.kappa       
