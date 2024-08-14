import numpy as np
from scipy.stats import kendalltau
from numpy.linalg import lstsq 

class gp_node:
    def __init__(self):
        self.symbol = None
        self.height = None
    def __str__(self):
        pass
    def eval(self,Xtrain):
        pass
    def calculate_kendall(self,Xtrain,Ytrain):
        Yp = self.eval(Xtrain)
        Y1 = Ytrain.astype(float)
        Y2 = Yp.astype(float)
        return kendalltau(Y1,Y2)[0] 
    def calculate_pearson(self,Xtrain,Ytrain):
        Yp = self.eval(Xtrain)
        Y1 = Ytrain.astype(float)
        Y2 = Yp.astype(float)
        return np.array(np.corrcoef(Y1,Y2))[0,1]

class gp_node_function(gp_node):
    def __init__(self,symbol,height):  
        self.symbol = symbol
        self.height = height
        self.wLeft = 1.
        self.wRight = 1.
        self.left = None
        self.right = None
    def __str__(self):
        return "("+self.left.__str__() + self.symbol + self.right.__str__()+")"
        #return "("+str(self.wLeft)+"*"+self.left.__str__() + self.symbol +str(self.wRight)+"*"+self.right.__str__()+")"

    def eval(self,Xtrain):
        if self.symbol=='+':
            return (self.wLeft*self.left.eval(Xtrain)) + (self.wRight*self.right.eval(Xtrain))
        elif self.symbol=='*':
            return self.wLeft*self.left.eval(Xtrain) * self.wRight*self.right.eval(Xtrain) 
        elif self.symbol=='/':
            return self.wLeft*self.left.eval(Xtrain) / self.wRight*self.right.eval(Xtrain) 
        elif self.symbol=='-':
            return self.wLeft*self.left.eval(Xtrain) - self.wRight*self.right.eval(Xtrain) 

class gp_node_terminal_variable(gp_node):
    def __init__(self,symbol,index):
        self.symbol = symbol
        self.index = index
        self.height = 1
    def __str__(self):
        return self.symbol
    def eval(self,Xtrain):
        return Xtrain[:,self.index]

class GeneticProgramming:
    def __init__(self,features_names,popsize=100,maxGenerations=100,probCrossover=1.,probMutation=1.,valFitness='Kendall',verbose=False):
        self.features_names = features_names
        self.nvariables = len(features_names)
        self.valFitness = valFitness
        self.popsize = popsize
        self.population1 = None
        self.population2 = None
        self.population3 = None
        self.fitness1 = None
        self.fitness2 = None
        self.fitness3 = None
        self.fitness1v = None
        self.fitness2v = None
        self.fitness3v = None
        self.MaxGenerations = maxGenerations
        self.functions = ['+','*','-','/']
        self.include_node_weights = True
        self.probCrossover = probCrossover
        self.probMutation = probMutation
        self.verbose = verbose
        self.indElite1,self.indElite2,self.indElite3 = None,None,None
        self.fitElite1,self.fitElite2,self.fitElite3 = 0,0,0

    ###########################################################################
    ####                    Init population                                   #
    ###########################################################################
    def __create_terminal_node(self,index=-1):
        if index==-1:
            index = np.random.randint(self.nvariables)
        symbol = 'x'+str(self.features_names[index])
        node = gp_node_terminal_variable( symbol,index )
        return node
        
    def __create_function_node(self,height):
        symbolFunction = np.random.choice(self.functions)
        node = gp_node_function( symbolFunction,height )
        if height == 2 :
            node.left = self.__create_terminal_node()
            node.right = self.__create_terminal_node()
        else:
            node.left = self.__create_function_node(2)
            node.right = self.__create_function_node(2)
        return node

    def __init_population(self): #-------------------------------------------
        self.fitness1 = np.zeros((self.nvariables))
        self.fitness2 = np.zeros((self.popsize))
        self.fitness3 = np.zeros((self.popsize))
        self.fitness1v = np.zeros((self.nvariables))
        self.fitness2v = np.zeros((self.popsize))
        self.fitness3v = np.zeros((self.popsize))
        
        self.population1 = []
        for i in range(self.nvariables):
            self.population1.append('a')
            self.population1[i] = self.__create_terminal_node(i)
        
        self.population2 = []
        for i in range(self.popsize):
            self.population2.append('a')
            self.population2[i] = self.__create_function_node(2)
        
        self.population3 = []
        for i in range(self.popsize):
            self.population3.append('a')
            self.population3[i] = self.__create_function_node(3)
        
        self.population1 = np.array(self.population1)
        self.population2 = np.array(self.population2)
        self.population3 = np.array(self.population3)

    ###########################################################################
    ####        Fitness and individuals                              #
    ###########################################################################
    def __calculate_fitness(self,ind,Xtrain,Ytrain):
        if self.valFitness=='Kendall':
            value = ind.calculate_kendall(Xtrain,Ytrain)
        else:
            value = ind.calculate_pearson(Xtrain,Ytrain)
        if np.isnan(value):
            return -1
        else:
            return np.abs(value)
            
    def __fitness_population(self,Xtrain,Ytrain,Xval,Yval): #-------------------------------------------
        for i in range(self.nvariables):
            self.fitness1[i] = self.__calculate_fitness(self.population1[i],Xtrain,Ytrain)
            self.fitness1v[i] = self.__calculate_fitness(self.population1[i],Xval,Yval)
        for i in range(self.popsize):
            self.fitness2[i] = self.__calculate_fitness(self.population2[i],Xtrain,Ytrain)
            self.fitness3[i] = self.__calculate_fitness(self.population3[i],Xtrain,Ytrain)
            self.fitness2v[i] = self.__calculate_fitness(self.population2[i],Xval,Yval)
            self.fitness3v[i] = self.__calculate_fitness(self.population3[i],Xval,Yval)

    ###########################################################################
    ####                   GP Operators                                       #
    ###########################################################################
  
    def __tournament(self,population,negative=False):
        if population==1:
            i1 = np.random.randint(self.nvariables)
            i2 = np.random.randint(self.nvariables)
        elif population==2 or population==3:
            i1 = np.random.randint(self.popsize)
            i2 = np.random.randint(self.popsize)

        if population==1:
            if not negative:
                index = i1 if self.fitness1[i1] >= self.fitness1[i2] else i2
            else:
                index = np.argmin(self.fitness1)

        elif population==2:
            if not negative:
                index = i1 if self.fitness2[i1] >= self.fitness2[i2] else i2
            else:
                index = np.argmin(self.fitness2)

        elif population==3:
            if not negative:
                index = i1 if self.fitness3[i1] >= self.fitness3[i2] else i2
            else:
                index = np.argmin(self.fitness3)

        return index

    def __deep_copy(self,node):
        if isinstance(node, gp_node_function):
            nnode = gp_node_function( node.symbol, node.height)
            nnode.wLeft = node.wLeft
            nnode.wRight = node.wRight
            nnode.left = self.__deep_copy( node.left )
            nnode.right = self.__deep_copy( node.right )
            return nnode
        elif isinstance(node,gp_node_terminal_variable):
            nnode = gp_node_terminal_variable( node.symbol,node.index )
            return nnode 

    def __mutation(self,node):  
        if isinstance(node,gp_node_function):
            if np.random.rand()<=0.5:
                node.symbol = np.random.choice(self.functions)
            else:
                if np.random.rand()<=0.5:
                    self.__mutation(node.left)
                else:
                    self.__mutation(node.right)
        else:
            index = np.random.randint(self.nvariables)
            symbol = 'x'+str(self.features_names[index]) 
            node.index = index
            node.symbol = symbol        
        return node

    def __crossover(self,parent1,parent2):
        par1 = self.__deep_copy(parent1)
        par2 = self.__deep_copy(parent2)
        if np.random.rand()<0.5:
            if np.random.rand()<=0.5:
                child1 = par1.left
                child2 = par2.left
                par1.left = child2
                par2.left = child1
            else:
                child1 = par1.left
                child2 = par2.right
                par1.left = child2
                par2.right = child1
        else:
            if np.random.rand()<=0.5:
                child1 = par1.right
                child2 = par2.left
                par1.right = child2
                par2.left = child1
            else:
                child1 = par1.right
                child2 = par2.right
                par1.right = child2
                par2.right = child1
        return par1,par2
        '''
        symbolFunction = np.random.choice(self.functions)
        offspring = gp_node_function( symbolFunction,height_parent+1 )
        offspring.left = self.__deep_copy(parent1)
        offspring.right = self.__deep_copy(parent2)
        return offspring
        '''

    def elite(self,verbose=True):
        i1 = np.argmax(self.fitness1v)
        i2 = np.argmax(self.fitness2v)
        i3 = np.argmax(self.fitness3v)

        if self.fitness1v[i1] >= self.fitElite1:
            self.indElite1,self.fitElite1 = self.__deep_copy(self.population1[i1]), self.fitness1v[i1]
        if self.fitness2v[i2] >= self.fitElite2:
            self.indElite2,self.fitElite2 = self.__deep_copy(self.population2[i2]), self.fitness2v[i2]
        if self.fitness3v[i3] >= self.fitElite3:
            self.indElite3,self.fitElite3 = self.__deep_copy(self.population3[i3]), self.fitness3v[i3]

    ###########################################################################
    ####                   Print population                                   #
    ###########################################################################
    def print_population(self):
        print(' ---------- Population ------------')
        for i in range(self.nvariables):
            print( i+1, self.fitness1[i],self.population1[i].height, self.population1[i] )
        for i in range(self.popsize):
            print( i+1, self.fitness2[i],self.population2[i].height, self.population2[i] )
        for i in range(self.popsize):
            print( i+1, self.fitness3[i],self.population3[i].height, self.population3[i] )

    def fit(self,Xtrain,Ytrain,Xval,Yval):
        self.__init_population()  

        self.__fitness_population(Xtrain,Ytrain,Xval,Yval)
        self.elite()
        
        i = 0
        while i <= self.MaxGenerations*self.popsize:
            i += 1
            p1 = self.__tournament(population=2)
            p2 = self.__tournament(population=2)
            if np.random.rand() <= self.probCrossover:
                parent1,parent2 = self.population2[p1], self.population2[p2]
                offspring1,offspring2 = self.__crossover(parent1,parent2)                
            else:
                offspring1 = self.__deep_copy(self.population2[p1])
                offspring2 = self.__deep_copy(self.population2[p2])
            if np.random.rand()<= self.probMutation:
                offspring1 = self.__mutation(offspring1)
                offspring2 = self.__mutation(offspring2)
            oFit1 = self.__calculate_fitness(offspring1,Xtrain,Ytrain)
            oFit2 = self.__calculate_fitness(offspring2,Xtrain,Ytrain)
            pos = self.__tournament(population=2,negative=True)
            if not np.isnan(oFit1) and oFit1>=oFit2:
                self.population2[pos] = offspring1
                self.fitness2[pos] = oFit1
                self.fitness2v[pos] = self.__calculate_fitness(offspring1,Xval,Yval)
            elif not np.isnan(oFit2) and oFit2>=oFit1:
                self.population2[pos] = offspring2
                self.fitness2[pos] = oFit2
                self.fitness2v[pos] = self.__calculate_fitness(offspring2,Xval,Yval)

            p1 = self.__tournament(population=3)
            p2 = self.__tournament(population=3)
            if np.random.rand() <= self.probCrossover:
                parent1,parent2 = self.population3[p1], self.population3[p2]
                offspring1,offspring2 = self.__crossover(parent1,parent2)                
            else:
                offspring1 = self.__deep_copy(self.population3[p1])
                offspring2 = self.__deep_copy(self.population3[p2])
            if np.random.rand()<= self.probMutation:
                offspring1 = self.__mutation(offspring1)
                offspring2 = self.__mutation(offspring2)
            oFit1 = self.__calculate_fitness(offspring1,Xtrain,Ytrain)
            oFit2 = self.__calculate_fitness(offspring2,Xtrain,Ytrain)
            pos = self.__tournament(population=3,negative=True)
            if not np.isnan(oFit1) and oFit1>=oFit2:
                self.population3[pos] = offspring1
                self.fitness3[pos] = oFit1
                self.fitness3v[pos] = self.__calculate_fitness(offspring1,Xval,Yval)
            elif not np.isnan(oFit2) and oFit2>=oFit1:
                self.population3[pos] = offspring2
                self.fitness3[pos] = oFit2
                self.fitness3v[pos] = self.__calculate_fitness(offspring2,Xval,Yval)
            
            self.elite()
            
            

        return self.indElite1,self.indElite2,self.indElite3

        
        