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
        ##########exemplo#############\n
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
            #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
            freq_classes=np.bincount(self.rotulos_treinamento[np.in1d(self.arrayDist[row],np.sort(self.arrayDist[row])[:self.k_init])][:self.k_init] )
            #print 'freq_classes: ',freq_classes
            k_modificado=self.k_init            
            controle=False
            control_taxa_acerto=[]
            classificacao_aux=[]
            control_Kmodificado=[]
            #numero de iteracoes
            cont=0
            while cont < self.max_iter and controle == False:
                #Controlar o total de iteracoes
                cont+=1
                #calcular a taxa de acerto da frequencia das classes
                value_taxa_acerto=np.max(freq_classes)/float(np.sum(freq_classes))
                #Controlar os valores das taxas de acertos
                control_taxa_acerto.append(value_taxa_acerto)
                if value_taxa_acerto >=    self.taxa_acerto:
                    control_Kmodificado.append(k_modificado)
                    #print 'maior que 0.5: '
                    #Selecionar a classe para a maximo k vizinhos
                    classificacao[row]=classes[np.in1d(freq_classes,np.max(freq_classes))][0]
                    valor_taxa_acerto[row]=max(control_taxa_acerto)
                    valor_KNN[row]=k_modificado
                    #Controle para quando satisfazer a taxa de acerto
                    controle=True       
                    break
               
                else:
                    #avaliar se max < sum*taxa
                    if  round((np.sum(freq_classes)*self.taxa_acerto)) > np.max(freq_classes):
                        #soma valor ao k_modificado
                        k_modificado=k_modificado+(round((np.sum(freq_classes)*self.taxa_acerto))-np.max(freq_classes))
                        #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
                        freq_classes=np.bincount(self.rotulos_treinamento[np.in1d(self.arrayDist[row],np.sort(self.arrayDist[row])[:k_modificado])][:k_modificado])
                        classificacao_aux.append(classes[np.in1d(freq_classes,np.max(freq_classes))][0])
                    else:
                        #soma valor ao k_modificado
                        k_modificado=k_modificado+round((np.max(freq_classes)*self.taxa_acerto))
                        #sort arrayDist, seleciona o k vizinhos e seleciona a classe que cada um representa nas classes de treinamento
                        freq_classes=np.bincount(self.rotulos_treinamento[np.in1d(self.arrayDist[row],np.sort(self.arrayDist[row])[:k_modificado])][:k_modificado])
                        classificacao_aux.append(classes[np.in1d(freq_classes,np.max(freq_classes))][0])
                    #-inserir k modificado na lista
                    control_Kmodificado.append(k_modificado)

                               
                #Inseri valor da classe na variavel classificacaocom maior taxa de acerto
                classificacao[row]= classificacao_aux[control_taxa_acerto.index(max(control_taxa_acerto))]                              
                valor_taxa_acerto[row]=max(control_taxa_acerto)
                valor_KNN[row]=control_Kmodificado[control_taxa_acerto.index(max(control_taxa_acerto))]
            #print 'freq_class: ',freq_classes                
            #print 'k atualizado: ',k_modificado
            #print 'classificacao AUX: ',classificacao_aux
            #print 'taxa acerto : ',control_taxa_acerto
            #print 'classe: ',classificacao[row]                                

        
        #Retorna um array com os rotulos
        return classificacao,valor_taxa_acerto,valor_KNN
'''#Dados treinamento
dados_trein=np.array([[10,20,30],[11,23,36],[16,29,35],[12,21,47]])
rotulos_trein = np.array([2,2,1,1])
#dados com novos exemplos a ser classificados
dados=np.array([[11,21,31],[12,22,32],[13,23,33],[15,25,45],[18,29,39],[9,20,29]])
knn = IterativoKNN(dados_trein,rotulos_trein,dados,1,1,0.7,3,1)
classificacao = knn.avaliarKNN()
print classificacao'''