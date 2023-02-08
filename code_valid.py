import numpy as np

class Tree():
    def __init__(self,data: np.ndarray,depth=None) -> None:
        self.data = data
        self.left= None
        self.right = None
        self.depth = depth if depth else 0 
        self.n = len(data)
        self.min_samples_split = 5 
        self.ymean = sum(data) / len(data)
        
    def __str__(self) -> str:
        return "Data: "+ str(self.data) +" |Len: " +  str(len(self.data)) + " |Mean:" + str(self.ymean) +" |SSR:"
    
    
    def get_all_nodes(self,n,nodes=[]):
        
        nodes = []

        if(n != None):
            nodes.append(n)
            self.get_all_nodes(n.left,nodes)
            self.get_all_nodes(n.right,nodes)
    
        return nodes
    
    def prune(self,toBePruned):

        if self == toBePruned:
            self.left = None
            self.right = None
            return

        if self.left != None and self.right != None:
            self.left.prune(toBePruned)
            self.right.prune(toBePruned)

    def get_ssr(self,childs,tree_count,to_be_pruned=None):
        """
        Returns sum of squarred residuals.
        """
        ssr = 0
        data_count = 0

        if len(childs) == 1:
            for y in childs[0].data:
                ssr += (y - childs[0].ymean)**2

        elif to_be_pruned == None or tree_count == 0 :

            for node in childs:
                if(node.left == None and node.right == None):
                    for y in node.data:
                        ssr += (y - node.ymean)**2
        else:
            #print(f"{tree_count}. SUBTREE")
            for node in childs:
                if((node != to_be_pruned.left and node != to_be_pruned.right)  and ((node.left not in childs and node.right not in childs) or node == to_be_pruned)):

                    s_ssr = 0
                    for y in node.data:
                        ssr += (y - node.ymean)**2
                        s_ssr += (y - node.ymean)**2
                    #print("LEN of DATA: "+ str(len(node.data)) +" |DATA: " + str(node.data) + " |MEAN: " + str(node.ymean) + " |SUB_SSR:" + str(s_ssr))
                    data_count += len(node.data)
        
        #print("Total Data Len of Leaf Nodes: ", data_count)            
        #print("SSR: ",ssr)
        tree_count += 1
        return ssr,tree_count

    
    def get_tree_scores(self): 

        self.nodes = self.get_all_nodes(self)
        print("Len nodes ",len(self.nodes))
        print("leaf count", self.get_leaf_count(self.nodes))
        #ssr_subtrees = dict()
        self.ssr_subtrees = []
        tree_count = 0
        
        while(len(self.nodes)>1):
            if tree_count > 20:
                break
            
            nodes_copy = self.nodes.copy()
            max_depth = 0

            for c in self.nodes:
                if((c.left in self.nodes) and (c.right in self.nodes) and (c.depth >= max_depth)):
                    max_depth = c.depth
                    the_node = c
                    
            #if len(self.nodes) == 2:
                #print(self.nodes[0].data,self.nodes[1].data)

            #print("\n//LEN of DATA: "+ str(len(the_node.data)) +" |DATA: " + str(the_node.data) + " |MEAN: " + str(the_node.ymean))
            #ssr_subtrees[str(the_node)],tree_count = self.get_ssr(nodes,tree_count,the_node)
            ssr,tree_count = self.get_ssr(self.nodes,tree_count,the_node)
            self.ssr_subtrees.append([ssr,nodes_copy])
            self.nodes[:] = [c for c in self.nodes if ((c!=the_node.left and c!=the_node.right) or c == the_node)]
            for i in self.nodes:
                print(len(i.data))
            print("\n")
        
        self.alphas = self.find_alpha(self.ssr_subtrees)
        #ssr_subtrees[str(root)],tree_count = self.get_ssr(nodes,tree_count)
        return self.alphas

    def find_alpha(self,ssr_subtrees):
        """
        ssr_subtrees: 2-d list [[ssr,nodes],[ssr,nodes],...]
        """
        tree_scores = []
        
        tree_scores.append([ssr_subtrees[0][1],ssr_subtrees[0][0],0])
        print("\nNumber of subtrees: ",len(ssr_subtrees))

        for idx,tree in enumerate(ssr_subtrees):

            if(idx == 0):
                continue
            
            alpha = 0
            leafs = self.get_leaf_count(tree[1]) 
            

            while(alpha != 200000):

                tree_score = tree[0] + alpha *  leafs
                success = 0

                for i in range(idx):

                    prev_leaf = self.get_leaf_count(ssr_subtrees[i][1])
                    prev_score = ssr_subtrees[i][0] + alpha * prev_leaf

                    if(tree_score > prev_score):
                        alpha += 1
                        break
                    else:
                        success += 1
                
                if success == idx:
                    #print(f"{idx}. treenin ssrı, leaf_countu, alpha değeri ve tree scoreu: ",tree[0],leafs,alpha, tree_score)
                    tree_scores.append([tree[1],tree[0],alpha])
                    break
                
            if(alpha == 200000):
                print("DOGRU ALPHA BULUNAMADI, tree index: ",idx) 

        return tree_scores

    def get_leaf_count(self,nodes):

        count = 0

        for node in nodes:
            if((node.left == None and node.right == None) or (node.left not in nodes and node.right not in nodes)):
                count += 1
        
        return count
    
    def inorder(self):

        if self.left != None and self.right != None: 
            print(self.data)
            self.left.inorder()
            self.right.inorder()
        elif(self != None):
            print(self.data)

    def build_tree(self,counter=1):
        

        if(self.n >= self.min_samples_split):
            
            left_data, right_data = self.data[self.data <= self.ymean],self.data[self.data > self.ymean]

            left = Tree(
                data=left_data,
                depth=self.depth + 1
                
            )
            self.left = left
            #print(f'{counter}. node {self.left.depth}. levelde olusturuldu')
            counter += 1
            counter = self.left.build_tree(counter)

            right = Tree(
                data=right_data,
                depth=self.depth + 1,
            )
            self.right = right
            #print(f'{counter}. node {self.right.depth}. levelde olusturuldu')
            counter += 1
            counter = self.right.build_tree(counter)

        return counter

        


# root = Tree([1,2,3,4,5,6,7,8,9],0)
# root.left = Tree([1,2,3,4],1)
# root.right = Tree([5,6,7,8,9],1)
# root.left.left = Tree([1,2],2)
# root.left.right = Tree([3,4],2)
# root.right.left = Tree([5,6],2)
# root.right.right = Tree([7,8,9],2)

data = np.random.randint(0,100,size=50)
data_2 = []
for idx,i in enumerate(data):
    if idx < 30:
        data_2.append(i)
data_2 = np.array(data_2)


root = Tree(data=data)
root_2 = Tree(data=data_2)

node_count = root.build_tree()
nc_2 = root_2.build_tree()

alphas = root.get_tree_scores()
alphas_2 = root_2.get_tree_scores()


