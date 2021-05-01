def get_unit_matrix(size):
    """
    :param size: integer that described the matrix's size- the matrix is n*n size
    :return: the unit matrix in the right size
    """
    unit_mat = [[0] * size for _ in range(size)]
    for i in range(size):
        unit_mat[i][i] = 1
    return unit_mat

def find_L_matrix(mat):
    l_mat= [[0] * len(mat) for _ in range(len(mat))]
    for i in range(len(l_mat)):
        for j in range(i):
            l_mat[i][j] = mat[i][j]
    return l_mat


def find_U_mat(mat):
    u_mat=[[0] * len(mat) for _ in range(len(mat))]
    for i in range(len(u_mat)):
        for j in range(i+1, len(mat)):
            u_mat[i][j] = mat[i][j]
    return u_mat

def find_D_mat(mat):
    d_mat = [[0] * len(mat) for _ in range(len(mat))]
    for i in range(len(d_mat)):
        for j in range(len(d_mat)):
            if j == i:
                d_mat[i][j] = mat[i][j]
    return d_mat


def print_matrix(A):
    """this func print that matrix
    :param A: matrix
    :return: no return value
    """
    print('\n'.join(['\t'.join(['{:4}'.format(item) for item in row])
                     for row in A]))


def multiply_matrices(mat1, mat2):
    """
    :param mat1: the first matrix
    :param mat2: the second matrix
    :return: multiply between matrices
    """
    if len(mat1[0]) != len(mat2):
        return None
    result_mat = [[0] * len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):  # rows
        for j in range(len(mat2[0])):  # cols
            for k in range(len(mat2)):
                result_mat[i][j] += (mat1[i][k] * mat2[k][j])
    return result_mat

def replace_line_in_matrix(mat, i):
    """ if the pivot is zero than we replace lines
    :param mat: matrix
    :param i: index
    :return: the updated mat
    """
    unit_mat = get_unit_matrix(len(mat))
    max_value_index = 0
    check = False
    for j in range(i + 1, len(mat)):
        if abs(mat[j][i]) > abs(mat[i][i]):
            max_value_index = j
            check = True
    if check:
        temp = unit_mat[i]
        unit_mat[i] = unit_mat[max_value_index]
        unit_mat[max_value_index] = temp
    return unit_mat

def compare_two_matrices(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: boolean value - true is mat1==mat2. else , false
    """
    if mat1 and mat2 is not None:
        for i in range(len(mat1)):
            for j in range(len(mat2)):
                if mat1[i][j] != mat2[i][j]:
                    return False
        return True
    else:
        return False


def elementary_matrix_U(mat):
    """
    this func find the elementary matrix in any level to find the reverse matrix
    :param mat:
    :return:
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat



def find_elementary_matrix(mat):
    """this func find the elementary matrix in any level in order to find the reverse matrix
    :param mat: matrix
    :return: elementary matrix
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat



def unit_diagonal(mat):
    unit_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        if mat[i][i] != 1:
            unit_mat[i][i] = 1 / mat[i][i]
            return unit_mat
    return unit_mat

def inverse_mat(mat):
    """ return the inverse mat with Gauss Elimination
    :param mat:matrix in size n*n
    :return: inverse matrix
    """
    unit_mat = get_unit_matrix(len(mat))  # build unit matrix
    all_elementary_mat = unit_mat  # deep copy
    for i in range(len(mat)):
        u_mat = replace_line_in_matrix(mat, i)  # pivoting
        mat = multiply_matrices(u_mat, mat)
        all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
        for j in range(i, len(mat)):
            if u_mat is not None or compare_two_matrices(u_mat, unit_mat):
                mat = multiply_matrices(u_mat, mat)
                all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
            u_mat = find_elementary_matrix(mat)
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            l_mat = elementary_matrix_U(mat)
            if l_mat is not None:
                mat = multiply_matrices(l_mat, mat)
                all_elementary_mat = multiply_matrices(l_mat, all_elementary_mat)
    for i in range(len(mat)):
        if mat[i][i] != 1:
            diagonal_mat = unit_diagonal(mat)
            mat = multiply_matrices(diagonal_mat, mat)
            all_elementary_mat = multiply_matrices(diagonal_mat, all_elementary_mat)
    return all_elementary_mat


def add_between_matrices(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: mat1 + mat2
    """
    if len(mat1) != len(mat1):
        return None
    result_mat = [[0] * len(mat1) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            result_mat[i][j] = mat1[i][j] + mat2[i][j]
    return result_mat

def copy_matrix(mat):
    """
    :param mat: matrix
    :return: deep copy of matrix
    """
    copy_mat = [[0] * len(mat[0]) for _ in range(len(mat))]  # creating new zero matrix
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            copy_mat[i][j] = mat[i][j]
    return copy_mat

def check_diagonal(mat):
    sum=0
    check= True
    for i in range(len(mat)):
        sum=0
        for j in range(len(mat)):
            if i != j:
                sum+= abs(mat[i][j])
        if(abs(mat[i][i])<sum):
           check= False

    if check:
        print("There is a dominant diagonal. The matrix converge")
        return True
    else:
        print("There is No a dominant diagonal. The matrix does not converge")
        return False


def minus_mat(mat):
    for i in range(len(mat)):
        for j in range(len(mat)):
            mat[i][j]= -1*mat[i][j]
    return mat


def Yaakobi(mat, b):
    print("~~~Yaakobi~~~")
    number_of_iteration =1# counter the number of itrations
    epsilon = 0.00001
    x, y,z= 0 , 0, 0
    x_r1= (b[0][0]-mat[0][1]*y-mat[0][2]*z)/mat[0][0]
    y_r1= (b[1][0]-mat[1][0]*x-mat[1][2]*z)/mat[1][1]
    if mat[2][2] == 0:
        z_r1 = (b[2][0]-mat[2][0]*x-mat[2][1]*y)/0.00001
    else:
        z_r1 = (b[2][0]-mat[2][0]*x-mat[2][1]*y)/mat[2][2]
    print("Iteration number:\t", number_of_iteration, "\tXr+1:\t", x_r1, "\tYr+1:\t", y_r1, "\tZr+1:\t", z_r1)
    while(abs(x_r1-x) > epsilon):
        if number_of_iteration<=99:#in case there is no dominac diagonal we want to limit the number of iteraions
            x= x_r1
            y=y_r1
            z= z_r1
            x_r1 = (b[0][0] - mat[0][1] * y - mat[0][2] * z) / mat[0][0]
            y_r1 = (b[1][0] - mat[1][0] * x - mat[1][2] * z) / mat[1][1]
            if mat[2][2] == 0:
                z_r1 = (b[2][0] - mat[2][0] * x - mat[2][1] * y) / 0.00001
            else:
                z_r1 = (b[2][0] - mat[2][0] * x - mat[2][1] * y) / mat[2][2]
            number_of_iteration += 1
            print("Iteration number:\t", number_of_iteration, "\tXr+1:\t", x_r1, "\tYr+1:\t", y_r1,
              "\tZr+1:\t", z_r1)
        else:
            break
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Total of iterations:\t", number_of_iteration)


def gauss_zaidel(mat,b):
    print("~~~Gauss~~~")
    number_of_iteration = 1
    epsilon = 0.00001
    x, y, z = 0, 0, 0
    x_r1 = (b[0][0] - mat[0][1] * y - mat[0][2] * z) / mat[0][0]
    y_r1 = (b[1][0] - mat[1][0] * x_r1 - mat[1][2] * z) / mat[1][1]
    if mat[2][2] == 0:
        z_r1 = (b[2][0] - mat[2][0] * x_r1 - mat[2][1] * y_r1) / 0.00001
    else:
        z_r1 = (b[2][0] - mat[2][0] * x_r1 - mat[2][1] * y_r1) / mat[2][2]
    print("Iteration number:\t", number_of_iteration, "\tXr+1:\t", x_r1, "\tYr+1:\t", y_r1,
          "\tZr+1:\t", z_r1)
    while (abs(x_r1 - x) > epsilon):
        if number_of_iteration<= 99:
            x = x_r1
            y = y_r1
            z = z_r1
            x_r1 = (b[0][0] - mat[0][1] * y - mat[0][2] * z) / mat[0][0]
            y_r1 = (b[1][0] - mat[1][0] * x_r1 - mat[1][2] * z) / mat[1][1]
            if mat[2][2] ==0 :
                z_r1 = (b[2][0] - mat[2][0] * x_r1 - mat[2][1] * y_r1) / 0.00001
            else:
                z_r1 = (b[2][0] - mat[2][0] * x_r1 - mat[2][1] * y_r1) / mat[2][2]


            number_of_iteration += 1
            print("Iteration number:\t", number_of_iteration, "\tXr+1:\t", x_r1, "\tYr+1:\t", y_r1, "\tZr+1:\t", z_r1)
        else:
            break
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Total of iterations:\t", number_of_iteration)
    return number_of_iteration


def main():

    A = [[0, 2, 0],
         [2, 10, 4],
         [0, 4, 5]]

    b= [[2], [ 6], [5]]

    print_matrix(A)
    check= check_diagonal(A)
    if check == True:
        x =int(input("Please choose 1-Gauss, else- Yaakobi"))
        if(x==1):
            gauss_zaidel(A, b)
        else:
             Yaakobi(A,b)
    else:
        y= int(input("Would you like to try to get the matrix to converge anyway? 1-Yes, else-No"))
        if y == 1:
            for i in range(len(A)):
                A= multiply_matrices(replace_line_in_matrix(A,i), A)
            x = int(input("Please choose 1-Gauss, else- Yaakobi"))
            if (x == 1):
                gauss_zaidel(A, b)
            else:
                Yaakobi(A, b)
        else:
            print("Goodbye!")




if __name__ == "__main__":
    main()