import numpy as np
import copy
import sys
import time
import matplotlib.pyplot as plt

def keySwitch(M, c, l):
    c_star = getBitVector(c, l)
    return M.dot(c_star)


def getRandomMatrix(row, col, bound):
    A = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            A[i][j] = np.random.randint(bound)
    return A


def getBitMatrix(S, l):
    S_star = list()
    for i in range(l):
        S_star.append(S*2**(l-i-1))
    S_star = np.array(S_star).transpose(1, 2, 0).reshape(len(S), len(S[0])*l)
    return S_star


def getSecretKey(T):
    assert(T.ndim == 2)
    I = np.eye(len(T))  # num rows
    return hCat(I, T)


def hCat(A, B):
    return np.concatenate((A, B), 1)


def vCat(A, B):
    return np.concatenate((A, B), 0)


def keySwitchMatrix(S, T, l, aBound, eBound):
    S_star = getBitMatrix(S, l)
    A = getRandomMatrix(T.shape[1], S_star.shape[1], aBound)
    E = getRandomMatrix(S_star.shape[0], S_star.shape[1], eBound)
    return vCat(S_star + E - T.dot(A), A)


def encrypt(T, x, w, l):
    return keySwitch(keySwitchMatrix(np.eye(len(x)), T, l, aBound=10, eBound=10), w * x, l)


def linearTransformClient(G, S, T, l):
    return keySwitchMatrix(G.dot(S), T, l, aBound=10, eBound=10)


def vectorize(M):
    ans = np.zeros((len(M) * len(M[0]), 1))
    for i in range(len(M)):
        for j in range(len(M[0])):
            ans[i * len(M[0]) + j][0] = M[i][j]
    return ans


def decrypt(S, c, w):
    Sc = S.dot(c)
    return (Sc / w).astype('float').round().astype('int')


def innerProdClient(T, l):
    S = getSecretKey(T)
    tvsts = vectorize(S.T.dot(S)).T
    return keySwitchMatrix(tvsts, T, l, aBound=10, eBound=10)


def innerProd(c1, c2, M, w, l):

    cc1 = np.zeros((len(c1), 1))
    for i in range(len(c1)):
        cc1[i][0] = c1[i]

    cc2 = np.zeros((1, len(c2)))
    for i in range(len(c2)):
        cc2[0][i] = c2[i]

    cc = vectorize(cc1.dot(cc2))

    bv = getBitVector((cc / w).round().astype('int64'), l)

    return M.dot(bv)


def one_way_encrypt_vector(vector, c_ones, M_keys, w, l, scaling_factor=1000):
    padded_vector = np.random.rand(len(vector)+1)
    padded_vector[0:len(vector)] = vector

    vec_len = len(padded_vector)

    M_temp = (M_keys[vec_len-2].T*padded_vector*scaling_factor / (vec_len-1)).T
    e_vector = innerProd(c_ones[vec_len-2], c_ones[vec_len-2], M_temp, w, l)
    return e_vector.astype('int')


def s_decrypt(vec, T_keys, w):
    return decrypt(getSecretKey(T_keys[len(vec)-2]), vec, w)


def add_vectors(x, y, scaling_factor=10000):
    return x + y


def transpose(syn1, M_onehot, v_onehot, w, l, scaling_factor):

    rows = len(syn1)
    cols = len(syn1[0]) - 1

    max_rc = max(rows, cols)

    syn1_c = list()
    for i in range(len(syn1)):
        tmp = np.zeros(max_rc+1)
        tmp[:len(syn1[i])] = syn1[i]
        syn1_c.append(tmp)

    syn1_c_transposed = list()

    for row_i in range(cols):
        syn1t_column = innerProd(
            syn1_c[0], v_onehot[max_rc-1][row_i], M_onehot[max_rc-1][0], w, l) / scaling_factor
        for col_i in range(rows-1):
            syn1t_column += innerProd(syn1_c[col_i+1], v_onehot[max_rc-1]
                                      [row_i], M_onehot[max_rc-1][col_i+1], w, l) / scaling_factor

        syn1_c_transposed.append(syn1t_column[0:rows+1])

    return syn1_c_transposed


def int2bin(x):
    s = list()
    # mod = 2
    while(x > 0):
        s.append(int(x % 2))
        x = int(x / 2)
    return np.array(list(reversed(s))).astype('int64')


def getBitVector(c, l):
    m = len(c)
    c_star = np.zeros(l * m, dtype='int64')
    for i in range(m):
        local_c = int(c[i])
        if(local_c < 0):
            local_c = -local_c
        b = int2bin(local_c)
        if(c[i] < 0):
            b *= -1
        if(c[i] == 0):
            b *= 0
#         try:
        c_star[(i * l) + (l-len(b)): (i+1) * l] += b
#         except:
#             print(len(b))
#             print(i)
#             print(len(c_star[(i * l) + (l-len(b)): (i+1) * l]))
    return c_star


def sigmoid(layer_2_c, H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor):
    out_rows = list()
    for position in range(len(layer_2_c)-1):

        M_position = M_onehot[len(layer_2_c)-2][0]

        layer_2_index_c = innerProd(layer_2_c, v_onehot[len(
            layer_2_c)-2][position], M_position, w, l) / scaling_factor

        x = layer_2_index_c
        x2 = innerProd(x, x, M_position, w, l) / scaling_factor
        x3 = innerProd(x, x2, M_position, w, l) / scaling_factor
        x5 = innerProd(x3, x2, M_position, w, l) / scaling_factor
        x7 = innerProd(x5, x2, M_position, w, l) / scaling_factor

        xs = copy.deepcopy(v_onehot[5][0])
        xs[1] = x[0]
        xs[2] = x2[0]
        xs[3] = x3[0]
        xs[4] = x5[0]
        xs[5] = x7[0]

        out = mat_mul_forward(
            xs, H_sigmoid[0:1], scaling_factor, M_onehot, w, l)
        out_rows.append(out)
    return transpose(out_rows, M_onehot, v_onehot, w, l, scaling_factor)[0]


def load_linear_transformation(syn0_text, T_keys, l, scaling_factor=1000):
    syn0_text *= scaling_factor
    return linearTransformClient(syn0_text.T, getSecretKey(T_keys[len(syn0_text)-1]), T_keys[len(syn0_text)-1], l)


def outer_product(x, y, M_onehot, v_onehot, onehot, w, l, scaling_factor):
    flip = False
    if(len(x) < len(y)):
        flip = True
        tmp = x
        x = y
        y = tmp

    y_matrix = list()

    for i in range(len(x)-1):
        y_matrix.append(y)

    y_matrix_transpose = transpose(
        y_matrix, M_onehot, v_onehot, w, l, scaling_factor)

    outer_result = list()
    for i in range(len(x)-1):
        outer_result.append(mat_mul_forward(
            x * onehot[len(x)-1][i], y_matrix_transpose, scaling_factor, M_onehot, w, l))

    if(flip):
        return transpose(outer_result, M_onehot, v_onehot, w, l, scaling_factor)

    return outer_result


def mat_mul_forward(layer_1, syn1, scaling_factor, M_onehot, w, l):

    input_dim = len(layer_1)
    output_dim = len(syn1)

    buff = np.zeros(max(output_dim+1, input_dim+1))
    buff[0:len(layer_1)] = layer_1
    layer_1_c = buff

    syn1_c = list()
    for i in range(len(syn1)):
        buff = np.zeros(max(output_dim+1, input_dim+1))
        buff[0:len(syn1[i])] = syn1[i]
        syn1_c.append(buff)

    layer_2 = innerProd(syn1_c[0], layer_1_c, M_onehot[len(
        layer_1_c) - 2][0], w, l) / float(scaling_factor)
    for i in range(len(syn1)-1):
        layer_2 += innerProd(syn1_c[i+1], layer_1_c,
                             M_onehot[len(layer_1_c) - 2][i+1], w, l) / float(scaling_factor)
    return layer_2[0:output_dim+1]


def elementwise_vector_mult(x, y, scaling_factor, M_onehot, onehot, v_onehot, w, l):

    y = [y]

    # one_minus_layer_1 = transpose(y)

    outer_result = list()
    for i in range(len(x)-1):
        outer_result.append(mat_mul_forward(
            x * onehot[len(x)-1][i], y, scaling_factor, M_onehot, w, l))

    return transpose(outer_result, M_onehot, v_onehot, w, l, scaling_factor)[0]


def neuralNetwork():

    # HAPPENS ON SECURE SERVER

    l = 100
    w = 2 ** 26

    aBound = 10
    tBound = 10
    eBound = 10

    max_dim = 10

    scaling_factor = 1000

    # keys
    T_keys = list()
    for i in range(max_dim):
        T_keys.append(np.random.rand(i+1, 1))

    # one way encryption transformation
    M_keys = list()
    for i in range(max_dim):
        M_keys.append(innerProdClient(T_keys[i], l))

    M_onehot = list()
    for h in range(max_dim):
        i = h+1
        buffered_eyes = list()
        for row in np.eye(i+1):
            buffer = np.ones(i+1)
            buffer[0:i+1] = row
            buffered_eyes.append((M_keys[i-1].T * buffer).T)
        M_onehot.append(buffered_eyes)
    c_ones = list()
    for i in range(max_dim):
        c_ones.append(encrypt(T_keys[i], np.ones(i+1), w, l).astype('int'))

    v_onehot = list()
    onehot = list()
    for i in range(max_dim):
        eyes = list()
        eyes_txt = list()
        for eye in np.eye(i+1):
            eyes_txt.append(eye)
            eyes.append(one_way_encrypt_vector(
                eye, c_ones, M_keys, w, l, scaling_factor))
        v_onehot.append(eyes)
        onehot.append(eyes_txt)

    H_sigmoid_txt = np.zeros((5, 5))

    H_sigmoid_txt[0][0] = 0.5
    H_sigmoid_txt[0][1] = 0.25
    H_sigmoid_txt[0][2] = -1/48.0
    H_sigmoid_txt[0][3] = 1/480.0
    H_sigmoid_txt[0][4] = -17/80640.0

    H_sigmoid = list()
    for row in H_sigmoid_txt:
        H_sigmoid.append(one_way_encrypt_vector(
            row, c_ones, M_keys, w, l, scaling_factor))

    #Neural network
    np.random.seed(1234)

    input_dataset = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    output_dataset = [[0], [1], [1], [0]]

    input_dim = 3
    hidden_dim = 4
    output_dim = 1
    alpha = 0.015

    # one way encrypt our training data using the public key (this can be done onsite)
    y = list()
    for i in range(4):
        y.append(one_way_encrypt_vector(
            output_dataset[i], c_ones, M_keys, w, l, scaling_factor))
    x = list()
    for i in range(4):
        x.append(one_way_encrypt_vector(
            input_dataset[i], c_ones, M_keys, w, l, scaling_factor))

    # generate our weight values
    syn0_t = (np.random.randn(input_dim, hidden_dim) * 0.2) - 0.1
    syn1_t = (np.random.randn(output_dim, hidden_dim) * 0.2) - 0.1

    # one-way encrypt our weight values
    syn0 = list()
    for row in syn0_t:
        syn0.append(one_way_encrypt_vector(row, c_ones, M_keys,
                                           w, l, scaling_factor).astype('int64'))
    syn1 = list()
    for row in syn1_t:
        syn1.append(one_way_encrypt_vector(row, c_ones, M_keys,
                                           w, l, scaling_factor).astype('int64'))
    listofLoss = []
    listofTime = []
    # begin training
    start = time.time()
    for iter in range(200):

        decrypted_error = 0
        encrypted_error = 0
        #Feed forward
        for row_i in range(4):

            if(row_i == 0):
                layer_1 = sigmoid(syn0[0], H_sigmoid,
                                  M_onehot, v_onehot, w, l, scaling_factor)
            elif(row_i == 1):
                layer_1 = sigmoid(
                    (syn0[0] + syn0[1])/2.0, H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)
            elif(row_i == 2):
                layer_1 = sigmoid(
                    (syn0[0] + syn0[2])/2.0, H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)
            else:
                layer_1 = sigmoid((syn0[0] + syn0[1] + syn0[2])/3.0,
                                  H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)

            layer_2 = (innerProd(syn1[0], layer_1, M_onehot[len(
                layer_1) - 2][0], w, l) / float(scaling_factor))[0:2]

            #error
            layer_2_delta = add_vectors(layer_2, -y[row_i])

            syn1_trans = transpose(
                syn1, M_onehot, v_onehot, w, l, scaling_factor)

            one_minus_layer_1 = [
                (scaling_factor * c_ones[len(layer_1) - 2]) - layer_1]

            #calculate derivative of sigmoid
            sigmoid_delta = elementwise_vector_mult(
                layer_1, one_minus_layer_1[0], scaling_factor, M_onehot, onehot, v_onehot, w, l)
            layer_1_delta_nosig = mat_mul_forward(
                layer_2_delta, syn1_trans, 1, M_onehot, w, l).astype('int64')
            layer_1_delta = elementwise_vector_mult(
                layer_1_delta_nosig, sigmoid_delta, scaling_factor, M_onehot, onehot, v_onehot, w, l) * alpha

            syn1_delta = np.array(outer_product(
                layer_2_delta, layer_1, M_onehot, v_onehot, onehot, w, l, scaling_factor)).astype('int64')

            # update weights
            syn1[0] -= np.array(syn1_delta[0] * alpha).astype('int64')

            syn0[0] -= (layer_1_delta).astype('int64')

            if(row_i == 1):
                syn0[1] -= (layer_1_delta).astype('int64')
            elif(row_i == 2):
                syn0[2] -= (layer_1_delta).astype('int64')
            elif(row_i == 3):
                syn0[1] -= (layer_1_delta).astype('int64')
                syn0[2] -= (layer_1_delta).astype('int64')

            encrypted_error += int(np.sum(np.abs(layer_2_delta)
                                          ) / scaling_factor)
            decrypted_error += np.sum(
                np.abs(s_decrypt(layer_2_delta, T_keys, w).astype('float')/scaling_factor))

        sys.stdout.write("\r Iter:" + str(iter) + " Encrypted Loss:" + str(encrypted_error) +
                         " Decrypted Loss:" + str(decrypted_error) + " Alpha:" + str(alpha))
        end = time.time()
        if (iter % 10 == 0):
            tim = end - start
            listofTime.append(tim)
            listofLoss.append(decrypted_error)
            print()

        # stop training when encrypted error reaches a certain level
        if(decrypted_error < 1.3):
            break
    
    print("\nFinal Prediction:")

    for row_i in range(4):

        if(row_i == 0):
            layer_1 = sigmoid(syn0[0], H_sigmoid, M_onehot,
                              v_onehot, w, l, scaling_factor)
        elif(row_i == 1):
            layer_1 = sigmoid(
                (syn0[0] + syn0[1])/2.0, H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)
        elif(row_i == 2):
            layer_1 = sigmoid(
                (syn0[0] + syn0[2])/2.0, H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)
        else:
            layer_1 = sigmoid((syn0[0] + syn0[1] + syn0[2])/3.0,
                              H_sigmoid, M_onehot, v_onehot, w, l, scaling_factor)

        layer_2 = (innerProd(syn1[0], layer_1, M_onehot[len(
            layer_1) - 2][0], w, l) / float(scaling_factor))[0:2]
        print("True Pred:" + str(output_dataset[row_i]) + " Encrypted Prediction:" + str(
            layer_2) + " Decrypted Prediction:" + str(s_decrypt(layer_2, T_keys, w) / scaling_factor))
        return listofLoss, listofTime

listofLoss = []
listofLoss, listofTime = neuralNetwork()
# print(listofLoss)
plt.xlabel("Time in seconds")
plt.ylabel("Loss")
plt.plot(listofTime, listofLoss)
plt.show()
