import torch


# vector mul vector
# mul 和 * 一致，按照元素位置进行相乘，不做降维
def vector_mul_vector():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    return x * y, torch.mul(x, y)


# print(vector_mul_vector())
# tensor([ 4, 10, 18]),

# vector mul number
# 向量和标量相乘，会进行broadcast进行广播
# mul 和 * 同等效果
def vector_mul_element():
    x = torch.tensor([1, 2, 3])
    y = 2
    return x * y
    # return torch.mul(x,y)


# print(vector_mul_element())
# tensor([2, 4, 6])


# vector matmul vector
# 所有列相乘之后相加即可；
# 向量matmul 点积按照维度相乘之后再相加，会进行降维处理
def vector_matmul_vector():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    return torch.matmul(x, y)


# vector_matmul_vector()
# tensor(32)


# matrix matmul matrix
# 确保条件：第一矩阵.列 = 第二矩阵.行（行列准则）
# 第一矩阵行决定输出矩阵行
# 第二矩阵列决定输出矩阵列
def matrix_matmul_matrix():
    # PS:matmul替代操作
    # 1.通过第二矩阵转置并进行mm操作得到同样结果
    x = torch.tensor([
        [1, 2],
        [4, 5],
        [4, 5],
        [6, 7]
    ])
    print(x)
    y = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    # print(y)
    print(y.T)
    # 转置之后 是通过
    return torch.matmul(x, y), torch.mm(x, y)


# print(matrix_matmul_matrix())
# tensor([[ 9, 12, 15],
#         [24, 33, 42],
#         [24, 33, 42],
#         [34, 47, 60]])

# vector matmul matrix
# 首先确保满足：行列准则
def vector_matmul_matrix():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([
        [7, 8, 1],
        [9, 10, 1],
        [11, 12, 1]
    ])
    # operation process:
    # 7 + 18 + 33 = 58
    # 8 + 20 + 36 = 64
    # 1 + 2  + 3  = 6
    return torch.matmul(x, y)


# print(vector_matmul_matrix())
# tensor([58, 64,  6])


# matrix matmul vector
# 规则：
# 1. 将vector 转换成为2位，列变行，[[7],[8],[9]]
# 2. 所有matrix列和vector列保持一致；
# 输出形状：vector 决定了一维行（不变），matrix 行决定列输出

# 简单准则： matrix matmul vector
# 1.确保matrix.列 = vector.列
# 2 matrix.行 决定了输出结果的列，行一定是一维
# 多维进行拍扁，成为一维多列
def matrix_matmul_vector():
    x = torch.tensor([
        [1, 2, 3, 5],
        [4, 5, 6, 5]
    ])
    y = torch.tensor([
        6, 7, 8, 9
    ])
    # 将[7,8,9] 转换为：
    # [[7],[8],[9]]
    # 1*7 + 2*8 + 3*9 = 50
    # 4*7 + 5*8 + 3*9 = 122
    return torch.matmul(x, y)


print(matrix_matmul_vector())
# tensor([ 89, 152])


# # 1.将y进行填充变成:y=[[7],[8]]，因为需要保持和x的二维大小相同，此时 y.shape = (3,1)
# # 2. 进行行列相乘，相加，即：
# #    1*10 + 2*11 = 32
# #    3*10 + 4*11 = 74
# #    5*10 + 6*11 = 116
# #    7*10 + 8*11 = 158
# # 生成数组[[32],[74],[116],[158]] = (4,1)，再去掉维度即[4]
# # 第一矩阵决定行，第二矩阵决定列的定义没有变化，只是进行维度扩充结算，之后再结果中进行了缩减；
# def matrix_vec_match():
#     x = torch.tensor([
#         [1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]
#     ])

#     y = torch.tensor([
#         10, 11
#     ])

#     # 1*7 + 2*8 = 23
#     # 4*7 + 5*8 = 68
#     return torch.matmul(x,y)
#     # tensor([ 44,  86, 128, 170])

# # print(matrix_vec_match())
# # tensor([ 32,  74, 116, 158])

# x = torch.tensor([
#     [1, 2],
#     [3, 4],
#     [5, 6],
#     [7, 8]
# ])

# y = torch.tensor([
#     10, 11
# ])
# print(y.T)

# y_1 = torch.tensor([
#     10, 11
# ])

# print(x @ y)
# print(x * y_1)