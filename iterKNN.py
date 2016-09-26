# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:34:56 2015

@author: ruiz
"""
import numpy as np
    
class InterativoKNN:
    '''Metodo de classificacao k-vizinhos mais proximos iterativo \n
        treinamento = ND array numpy\n
        rotulos_treinamento= 1D numpy\n
        novos_exemplos = ND array numpy\n
        peso = 1D numpy or integer\n
        potencia = integer \n
        taxa_acerto = real (0-1)\n
        max_iter = integer \n
        k_init = integer
        ##########exemplo################\n
        #Dados treinamento
        dados_trein=np.array([[10,20,30],[11,23,36],[16,29,35],[12,21,47]])\n
        rotulos_trein = np.array([2,2,1,1])\n
        #dados com novos exemplos a ser classificados
        dados=np.array([[11,21,31],[12,22,32],[13,23,33],[15,25,45],[18,29,39],[9,20,29]])\n
        knn = IterativoKNN(dados_trein,rotulos_trein,dados,1,2,0.8,5,2)\n
        classificacao = knn.avaliarKNN()\n
        print classificacao
        >>>  [2 2 2 1 1 2]'''
    def __init__(self,treinamento,rotulos_treinamento,novos_exemplos,peso,potencia,taxa_acerto,max_iter,k_init):
        #iniciar variaveis
        self.treinamento = treinamento #ND array numpy
        self.rotulos_treinamento=rotulos_treinamento#1D numpy
        self.novos_exemplos = novos_exemplos #ND array numpy
        self.peso = peso #inteiro
        self.potencia = potencia #inteiro 
        self.taxa_acerto = taxa_acerto #real (0-1)
        self.max_iter = max_iter #inteiro
        self.k_init = k_init #inteiro
    def calcDistMinkowski(self):    
        '''Calcular a distancia entre treinamento e novos exemplos a ser classificados'''
        #criar um array colunas igual ao tamanho do treinamento e linhas igual aos novos exemplos  
        self.arrayDist=np.arange(self.treinamento.shape[0]*self.novos_exemplos.shape[0],dtype=np.float32).reshape(self.novos_exemplos.shape[0],self.treinamento.shape[0])
        
        #Percorrer array novos exemplos
        for i in xrange(self.novos_exemplos.shape[0]):
            #calcular distancia         
            self.arrayDist[i,:] = np.sum(self.peso*(abs(np.asarray(self.novos_exemplos[i,:],dtype=np.float32)-self.treinamento)**self.potencia),1)**(1./self.potencia)
        #print self.arrayDist
        #total de linhas do DIST deve ser igual total de linhas do novos exemplos
        assert self.arrayDist.shape[0]== self.novos_exemplos.shape[0]
        #Total de colunas dos dados treinamento deve ser igual as colunas do DIST
        assert self.arrayDist.shape[1]== self.treinamento.shape[0]
    def avaliarKNN (self):
        #criar array de distancias (similaridade)
        self.calcDistMinkowski()                            
        #Obter as classes:
        classes=np.unique(self.rotulos_treinamento)
        #Inserir o valor 0 como primeiro
        classes=np.insert(classes,0,0)
        #Criar o vetor array 1D
        classificacao = np.arange(self.novos_exemplos.shape[0])
        #numero de vizinhos
        valor_KNN = np.arange(self.novos_exemplos.shape[0])
        #taxa de acerto final
        valor_taxa_acerto = np.zeros(self.novos_exemplos.shape[0])
   
        #Loop sobre linhas novos_exemplos ou sobre o numero lintas do arrayDist
        for row in xrange(self.novos_exemplos.shape[0]): 
            #print 'instancia: ',row
            #variaveis aux
            control_taxa_acerto=[]
            #print 'freq_classes: ',freq_classes
            k_atualizado=self.k_init 
            #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
            freq_classes=np.bincount(self.rotulos_treinamento[np.in1d(self.arrayDist[row],np.sort(self.arrayDist[row])[:self.k_init])][:self.k_init] )
            #calcular a taxa de acerto da classe mais frequente
            value_taxa_acerto=np.max(freq_classes)/float(np.sum(freq_classes))
            #Controlar os valores das taxas de acertos
            control_taxa_acerto.append(value_taxa_acerto)
            #numero de iteracoes
            cont=1
            while cont < self.max_iter:
                #print 'iteracao: ',cont  
                #print 'k: ',k_atualizado
                #print 'Frequencia das classes: ',freq_classes
                #print 'Taxa de acerto: ',value_taxa_acerto
                
                #avalia a taxa de acerto
                if value_taxa_acerto >= self.taxa_acerto:
                        
                        classificacao[row]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]
                        valor_taxa_acerto[row]=np.max(freq_classes)/float(k_atualizado)
                        valor_KNN[row]=k_atualizado
                        #print 'k: ',k_atualizado
                        #print 'Frequencia das classes: ',freq_classes
                        #print 'Taxa de acerto: ',value_taxa_acerto
                        #print 'break'
                        break
                        
                   
                
                classificacao[row]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]    
                valor_taxa_acerto[row]=np.max(freq_classes)/float(k_atualizado)
                valor_KNN[row]=k_atualizado
                #calcular o valor que a proxima classe mais frequente devera obter
                freq_N1 = round((self.taxa_acerto*(k_atualizado-np.max(freq_classes)))/(1-self.taxa_acerto))
                #calcular o proximo valor K
                k_atualizado=(k_atualizado-np.max(freq_classes))+ freq_N1       
                #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
                freq_classes=np.bincount(self.rotulos_treinamento[np.in1d(self.arrayDist[row],np.sort(self.arrayDist[row])[:k_atualizado])][:k_atualizado] )
                #calcular a taxa de acerto da classe mais frequente
                value_taxa_acerto=np.max(freq_classes)/float(k_atualizado)#float(np.sum(freq_classes))
                #Controlar o total de iteracoes
                cont+=1
        
        #Retorna um array com os rotulos
        return classificacao,valor_taxa_acerto,valor_KNN
'''#Dados treinamento
dados_trein=np.array([[13,21,33],[13,21,32],[10,20,30],[11,23,36],[16,29,35],[12,21,47],[22,32,47]])
rotulos_trein = np.array([2,2,2,2,1,1,1])
#dados com novos exemplos a ser classificados
dados=np.array([[11,21,31],[12,22,32],[13,23,33],[15,25,45],[18,29,39],[9,20,29]])
classes_dados=np.array([2,2,2,1,1,2])
knn = InterativoKNN(dados_trein,rotulos_trein,dados,1,1,0.75,3,3)
classificacao = knn.avaliarKNN()
print classificacao'''