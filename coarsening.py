import numpy as np
import scipy.sparse


def coarsen(A, levels, self_connections=False):
    # #A是邻接矩阵，levels粗化了的图的数目（0），self_connection=false
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    """
    graphs, parents = metis(A, levels)#    Coarsen a graph multiple times using the METIS algorithm.
    print('def coarsen ret of func metis--graphs = {},parents = {}'.format(graphs,parents))
    perms = compute_perm(parents) #perms是新顺序
    # Return a list of indices to reorder the adjacency and data matrices so
    # that the union of two neighbors from layer to layer forms a binary tree.

    for i, A in enumerate(graphs):
        M, M = A.shape #包含了假节点

        if not self_connections:
            A = A.tocoo()#Convert this matrix to COOrdinate format.转为坐标格式
            A.setdiag(0)#        Set diagonal or off-diagonal elements of the array.


        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr() #csr格式是怎样
        A.eliminate_zeros() #移除0，为什么反而A会表示出值为0的坐标了
        graphs[i] = A#A的值给graphs了

        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

    return graphs, perms[0] if levels > 0 else None


def metis(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))#洗牌 0 - rid-1
    parents = [] #list
    degree = W.sum(axis=0) - W.diagonal() #W.sum(axis=0) 是每一列元素相加，得到一个一维的数组,W.diagonal()得到对角矩阵，只表示对角线的一维矩阵
    graphs = []
    graphs.append(W)#list, N*N sparse matrix
    #supernode_size = np.ones(N)
    #nd_sz = [supernode_size]
    #count = 0

    #while N > maxsize:
    for _ in range(levels):

        #count += 1

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()#去掉单维

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR#将W的行索引排序，存储在rr中，对应的列和值在cc,vv中
        idx_row, idx_col, val = scipy.sparse.find(W)#返回非零元素索引和值
        perm = np.argsort(idx_row)#对idx_row的元素排序，返回新的数组perm，perm里的值是idx_row的索引
        rr = idx_row[perm]#rr是行索引从小到大的数组
        cc = idx_col[perm]#是rr对应的列索引的从小到大的数组
        vv = val[perm]#上述对应的值
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
        parents.append(cluster_id)

        # TO DO
        # COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE 
        #supernode_size = full(   sparse(cluster_id,  ones(N,1) , supernode_size )     )
        #print(cluster_id)
        #print(supernode_size)
        #nd_sz{count+1}=supernode_size;

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        #degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):

    nnz = rr.shape[0] #shape[0] 行数
    N = rr[nnz-1] + 1#rr最后一个元素加1,

    marked = np.zeros(N, np.bool)#返回一个数组 ，初始化为False，个数为N
    rowstart = np.zeros(N, np.int32)#返回一个数组 ，初始化为0，个数为N
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    # 生成rowlength, rowstart
    #rowstart是存储从第几个元素开始换行,eg[0 3 5],元素个数是rr中存储的最后一个值加1，i.e. 2+1=3
    for ii in range(nnz):# rr中存储的是索引值，类似0 0 0 1 1 2 2 ，rowlength存储每一行有几个元素，但第一行会多加一个，最后一行会减一个，eg[4 2 1]
        rowlength[count] = rowlength[count] + 1#将当前位置元素加一
        if rr[ii] > oldval:#直到，rr中存储的索引大于oldval
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    # 遍历每一行？？在图上是什么意思呢
    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:# 如果tid没有被Mark，就先Mark  判断该行是否Mark
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):#按行遍历,如果没有被Mark，就计算两者间的 权值？，然后在该行中找 让权值最大的那个节点，即bestneighbor
                # 遍历tid对应的那一行的节点，tid决定哪一行，range(rowlength[tid])是该行的节点个数
                nid = cc[rs+jj]#找列索引，rs+jj,i.e. 行开始的标号+行偏移
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True # 标记最终选出来的bestneighbor

            clustercount += 1

    return cluster_id

def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:
        #print('parent: {}'.format(parent))

        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            #print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singelton.
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
                #print('new singelton: {}'.format(indices_node))
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
                #print('singelton childrens: {}'.format(indices_node))

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks合理检查
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]#stride设置为-1 是将元素顺序逆转

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
        == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])

def perm_data(x, indices):
    #序列改变
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:,i] = x[:,j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:,i] = np.zeros(N)
    return xnew

def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # 将新添加的假节点 放在图A后面
    # Add Mnew - M isolated vertices.
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # 将row和col重新随机排序
    # Permute the rows and the columns.
    perm = np.argsort(indices)#返回的是索引，索引指向原数组，对应的原数组元素有序排列
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A
