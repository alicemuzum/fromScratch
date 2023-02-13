import numpy as np
import tensorflow  as tf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
np.set_printoptions(precision=None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from tqdm import tqdm
 

class Node():
    def __init__(
        self,
        X: pd.DataFrame,
        Y: list,
        childs: list,
        min_samples_split = None,
        max_depth = None,
        depth=None,
        node_type=None,
        rule=None
    ):
        self.X = X
        self.Y = Y
        
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        self.depth = depth if depth else 0

        self.features = list(self.X.columns)

        self.node_type = node_type  if node_type else 'root'

        self.rule = rule if rule else ""

        self.ymean = np.mean(Y)
        self.residuals = self.Y - self.ymean
        self.mse = self.get_mse(Y,self.ymean)

        self.n = len(Y)

        self.left = None
        self.right = None

        self.best_feature = None 
        self.best_value=None

    def __str__(self) -> str:
        return self.depth

    @staticmethod
    def get_mse(y, yhat) -> float:
        """
        Method for calculation of mean squared error
        """

        n = len(y)

        r = y - yhat
        r = r**2
        r = np.sum(r)

        return r / n
    
    @staticmethod
    def ma(x:np.array, window: int) -> np.array:
        """
        Moving Average
        """
        return np.convolve(x, np.ones(window), 'valid') / window 
    

    def best_split(self) -> tuple:
        """
        Calculates best split for a decision tree

        """
        df = self.X.copy()
        df['Y'] = self.Y

        mse_base = self.mse

        #max_gain = 0

        best_feature = None
        best_value = None        
        
        for feature in self.features:
            
            # Drop NA values in dataframe  and sort rows by ascending order.
            Xdf = df.dropna().sort_values(feature)

            
            # ma() returns average value of every desired amount of consecutive numbers. In this case its every two consecutive number.
            # input = [ 20  30  40  45  50  60  70  75  80  85  90 120 160 180 190]
            # output = [ 25.   35.   42.5  47.5  55.   65.   72.5  77.5  82.5  87.5 105.  140. 170.  185. ]
            xmeans = self.ma(Xdf[feature].unique(),2)
            
    
            for value in xmeans:

                left_y = Xdf[Xdf[feature]<value]['Y'].values
                right_y = Xdf[Xdf[feature]>=value]['Y'].values

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    left_mean = np.mean(left_y)
                    right_mean = np.mean(right_y)

                res_left = left_y - left_mean
                res_right = right_y - right_mean
                
          
                # Axis = None means flatten matrices before concatenation.
                r = np.concatenate((res_left,res_right),axis=None)
                
            
                n = len(r)
                r = r ** 2 
                r = np.sum(r)
                mse_split = r / n

                # first mse value calculated while initialization, now we check if current node has lower mse.
                if mse_split < mse_base:
                    best_feature = feature
                    best_value=value

                    mse_base = mse_split
            
        return (best_feature,best_value)

    def build_tree(self,counter = 0):
        """
        Recursive method to create the decision tree
        """
        df = self.X.copy()
        df['Y'] = self.Y
        
        
        if(self.n >= self.min_samples_split):
            
            
            best_feature, best_value = self.best_split()

            if best_feature is not None:

                self.best_feature = best_feature
                self.best_value = best_value

                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature] > best_value].copy()
                
                left = Node(
                    left_df[self.features],
                    left_df['Y'].values.tolist(),
                    childs= self.childs,  
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='left_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.left = left
                print(f'{counter}. node {self.left.depth}. levelde olusturuldu')
                counter += 1
                counter = self.left.build_tree(counter)

                right = Node(
                    right_df[self.features],
                    right_df['Y'].values.tolist(),
                    childs= self.childs,   
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.right = right
                print(f'{counter}. node {self.right.depth}. levelde olusturuldu')
                counter += 1
                counter = self.right.build_tree(counter)
            
        return counter
                

    
    def get_all_nodes(self,nodes=[]):
    
        if(self != None):
            nodes.append(self)
            Node.get_all_nodes(self.left,nodes)
            Node.get_all_nodes(self.right,nodes)
    
        return nodes

    def get_ssr(self,nodes,ssr_counter,to_be_pruned=None):
        """
        childs(list): List of all the nodes of the tree.
        to_be_pruned(Node): The node to be used to decide which
        """
        ssr = 0
        data_count = 0

        if len(nodes) == 1:
            for y in nodes[0].Y:
                ssr += (y - nodes[0].ymean)**2

        elif(to_be_pruned == None or ssr_counter == 0):
            for node in nodes:
                if(node.right == None and node.left == None):
                    for y in node.Y:
                        ssr += (y - node.ymean)**2
        else:

            # How can I remember previous to_be_pruned is now a leaf node if I don't reassign its left and right node to None?
            # 
            for node in nodes:
                if((node != to_be_pruned.left and node != to_be_pruned.right) and ((node.left not in nodes and node.right not in nodes) or node == to_be_pruned)):
                    for y in node.Y:
                        #print(y,"-",node.ymean,"=",y-node.ymean)
                        ssr += (y - node.ymean)**2
                        data_count += 1

        print(f"{ssr_counter}. budamadaki chlids uzunlugu : {len(nodes)}  ve yapraklardaki toplam data: {data_count} ve gelen result: {ssr}")
        ssr_counter += 1

        return ssr, ssr_counter

    def get_tree_scores(self): 

        nodes = self.get_all_nodes()
        
        ssr_subtrees = []
        ssr_counter = 0
        alphas = []

        while(len(nodes)>1):

            nodes_copy = nodes.copy()

            max_depth = 0
            for c in nodes:
                if((c.left in nodes) and (c.right in nodes) and (c.depth >= max_depth)):
                    max_depth = c.depth
                    the_node = c
            
                        
            ssr_result, ssr_counter = self.get_ssr(nodes,ssr_counter,the_node)
            ssr_subtrees.append([ssr_result,nodes_copy])
            nodes[:] = [c for c in nodes if ((c!=the_node.left and c!=the_node.right) or c == the_node)]
        

        return ssr_subtrees
        
    def find_alpha(self,ssr_subtrees,alphas) -> int:
        """
        ssr_subtrees: 2-d list [[ssr,nodes],[ssr,nodes],...]
        """
        tree_scores = []
        

        if not alphas:
            tree_scores.append(ssr_subtrees[0][0])
        else:
            
            for tree in ssr_subtrees:

                prev_score = tree[0]
                alpha = 0
                tree_score = np.inf

                while(tree_score > prev_score):
                    prev_score = tree_score
                    tree_score = tree[0][0] + alpha * self.get_leaf_count(tree[0][1])   
                    alpha += 1
                
                tree_scores.append(tree[1],tree_score)

    def get_leaf_count(self,nodes):

        count = 0

        for node in nodes:
            if((node.left == None and node.right == None) or (node.left not in nodes and node.right not in nodes)):
                count += 1
        
        return count
    

    def predict(self, data_row: pd.DataFrame):
        """
        Returns the predicted value for given dataframe row.
        Note that you should only pass one single row at a time to this function.

        data: (pd.DataFrame)
        """    
        
        if self.best_feature is not None:


            if data_row.loc[self.best_feature] <= self.best_value:

                return self.left.predict(data_row)
            
            elif data_row.loc[self.best_feature] > self.best_value:

                return self.right.predict(data_row)
        
        else:
            
            return self.ymean   

    def test(self, data: pd.DataFrame, is_training=True):

        """
        Returns an array with predicted y values in the same order with input.
        """
       
        data = data.drop("SalePrice",axis=1)
        
        yh = []

        for index, row in data.iterrows():
            yh.append(self.predict(row))
        
        
        return np.array(yh)

    def print_info(self, width = 4):

        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}") 

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    def plot_results(self,Y,Yhat):
        """
        Plots line plot of predictions vs real values
        """
        x = list(range(len(Y)))
        loss = np.abs(Yhat - Y)
        plt.scatter(x,loss,s=1)
        plt.show()

    


            

#if __name__ == 'main':
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


train = pd.read_pickle("../data/final_train.pkl")

Y_train = train['SalePrice'].values
Y_train = Y_train / 100000
X_train = train.drop('SalePrice',axis=1)

childs = []
root = Node(X_train, Y_train,childs=childs)
root.build_tree()
root.get_tree_scores()

# test = pd.read_pickle("../data/final_test.pkl")
# check = root.test(train)

# root.plot_results(train['SalePrice'].to_numpy(),check)


# for idx,i in enumerate(check):
#     if idx == 5:
#         break
#     print(i,Y_train[idx])
    

