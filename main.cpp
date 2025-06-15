
/*
Enter number of equations 3
Enter number of variables 3


 Enter coefficient of equations 
 
 For ROW 1
 		Enter the coefficient X11:  2
		Enter the coefficient X12:  1
		Enter the coefficient X13:  1
		Enter the coefficient of RHS1:   180

 For ROW 2
 		Enter the coefficient X21:  1
		Enter the coefficient X22:  3
		Enter the coefficient X23:  2
		Enter the coefficient of RHS2:   300

 For ROW 3
 		Enter the coefficient X31:  2
		Enter the coefficient X32:  1
		Enter the coefficient X33:  2
		Enter the coefficient of RHS3:   240

 Enter coefficients of the objective function
		Enter the coefficient X41:  6
		Enter the coefficient X42:  5
		Enter the coefficient X43:  4


	initial array(Not optimal)
	2 1 1 1 0 0 
	1 3 2 0 1 0 
	2 1 2 0 0 1 

	 
	final array(Optimal solution)
	1 0 0.2 0.6 -0.2 0 
	0 1 0.6 -0.2 0.4 0 
	0 0 1 -1 0 1 

	Answers for the Constraints of variables(variables array)
	variable1: 48
	variable2: 84
	variable3: 0

	maximum value: 708

-------------------------------

   Example qution:    maximize -->  6x + 5y + 4z

		      Subject to  2x + y + z <= 180    slack varibles  2x + y + z + s1 = 180  
				  x + 3y + 2z <= 300                   x + 3y + 2z +s2 = 300
				  2x + y + 2z <= 240                    2x + y + 2z +s3 = 240

  you can make a arryas like below from this quetion:
        colSizeA = 6 // input colmn size
	rowSizeA = 3  // input row size
    float C[N]={-6,-5,-4,0,0,0};  //Initialize the C array  with the coefficients of the constraints of the objective function
    float B[M]={180,300,240};//Initialize the B array constants of the constraints respectively
*/	

/*
Simplex algorithm
https://github.com/chashikajw/simplex-algorithm
https://www.yogeshsn.com.np/references/programming/simplexc/
https://github.com/PetarV-/Algorithms/blob/master/Mathematical%20Algorithms/Simplex%20Algorithm.cpp
https://www.geeksforgeeks.org/python/simplex-algorithm-tabular-method/
https://pacha.dev/blog/2023/07/18/cpp11-simplex/index.html
https://www.alglib.net/linear-programming/simplex-method.php
https://cplusplus.algorithmexamples.com/web/Mathematical/Simplex%20Algorithm.html
https://math.libretexts.org/Bookshelves/Applied_Mathematics/Applied_Finite_Mathematics_(Sekhon_and_Bloom)/04%3A_Linear_Programming_The_Simplex_Method/4.02%3A_Maximization_By_The_Simplex_Method
https://medium.com/@muditbits/simplex-method-for-linear-programming-1f88fc981f50
https://www.r-bloggers.com/2023/07/a-naive-simplex-phase-2-implementation-with-c-11-and-r/
*/	

/*

The main method is in this program itself.

Instructions for compiling=>>

Run on any gcc compiler=>>

Special***** should compile in -std=c++11 or C++14 -std=gnu++11  *********  (mat be other versions syntacs can be different)

turorials point online compiler
==> go ti link   http://cpp.sh/ or  https://www.tutorialspoint.com/cplusplus/index.htm and click try it(scorel below) and after go to c++ editor copy code and paste.
after that click button execute.

if you have -std=c++11 you can run in command line;
g++ -o output Simplex.cpp
./output


How to give inputs to the program =>>>

   Example:
    colSizeA = 6 // input colmn size
    rowSizeA = 3  // input row size

    float C[N]={-6,-5,-4,0,0,0};  //Initialize the C array  with the coefficients of the constraints of the objective function
    float B[M]={240,360,300};//Initialize the B array constants of the constraints respectively


   //initialize the A array by giving all the coefficients of all the variables
   float A[M][N] =  {
                 { 2,  1,  1,   1,  0, 0},
                { 1,  3,  2,   0,  1, 0 },
                {   2,    1,  2,   0,  0,  1}
                };

*/


//----------------------------------

#include <stdio.h>

//----------------------------------

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>

//----------------------------------

#include <iostream>
#include <cmath>
#include <vector>

//----------------------------------

namespace example0 {
int main_example0() {
const int DDD = 1;
    const int a = 3 + DDD; /* number of equations */
    const int b = 3 + DDD; /* number of variables */
    int i, j, k, pc, pr, count;
    float pe, min_ratio, ratio, pc_check, store;

    float table[a+DDD][b+DDD];
    float rhs[a+DDD];
    float obj[a+DDD];
    int row = a + 1+DDD;
    int col = a + b + 1+DDD;
    float OptTable[row+DDD][col+DDD];

    /* coefficients of equations: row 1->=a, coef 1->=b */
    // a
    table[1][1] = 2; // b
    table[1][2] = 1; // b
    table[1][3] = 1; // b
    // + rhs
    
    table[2][1] = 1;
    table[2][2] = 3;
    table[2][3] = 2;
    
    table[3][1] = 2;
    table[3][2] = 1;
    table[3][3] = 2;

    /* the coefficient of RHS: 1->=a */
    rhs[1] = 180;
    rhs[2] = 300;
    rhs[3] = 240;

    /* coefficients of the objective function: 1->=b */
    obj[1] = 6; // 41
    obj[2] = 5; // 42
    obj[3] = 4; // 43

    for (i = 1; i < row; ++i) {
        for (j = 1; j <= col; ++j) {
            if (j <= b) {
                OptTable[i][j] = table[i][j];
            }
            if (j > b) {
                if (j == b + i) {
                    OptTable[i][j] = 1.0;
                } else {
                    OptTable[i][j] = 0.0;
                }
            }
            if (j == col) {
                OptTable[i][col] = rhs[i];
            }
        }
    }
    for (j = 1; j <= col; ++j) {
        if (j <= b) {
            OptTable[row][j] = -obj[j];
        }
        if (j > b) {
            OptTable[row][j]

 = 0.0;
        }
    }

    while (1) {
        for (k = 1; k <= col; ++k) {
            printf("\n Iteration %d \n ", k - 1);
            for (j = 1; j < col; j++) {
                printf("\t X%d", j);
            }
            printf("\t RHS \n");
            for (i = 1; i <= row; ++i) {
                for (j = 1; j <= col; ++j) {
                    printf("\t%0.2f", OptTable[i][j]);
                }
                printf("\n");
            }

            pc = 1, pc_check = 0, count = 0;
            for (j = 1; j <= col; ++j) {
                if (OptTable[row][j] < 0) {
                    if (pc_check > OptTable[row][j]) {
                        pc_check = OptTable[row][j];
                        pc = j;
                        count = 1;
                    }
                }
                if (j == col && count == 0) {
                    goto successed;
                }
            }

            pr = 0, pe = 1, min_ratio = 1;
            for (i = 1; i <= a; ++i) {
                if (OptTable[i][pc] > 0) {
                    ratio = OptTable[i][col] / OptTable[i][pc];
                    if (pr == 0) {
                        min_ratio = OptTable[i][col] / OptTable[i][pc];
                        ++count;
                        pe = OptTable[i][pc];
                        pr = i;
                    }
                    if (min_ratio >= ratio && pr != 0) {
                        if (min_ratio == ratio) {
                            if (OptTable[i][pc] < pe) {
                                pe = OptTable[i][pc];
                                pr = i;
                            }
                        } else {
                            min_ratio = ratio;
                            pr = i;
                            pe = OptTable[i][pc];
                        }
                    }
                }
                if (OptTable[i][pc] < 0 && i == a) {
                    printf("\n\tUnbounded solution\n");
                    return 0;
                }
            }

            for (j = 1; j <= col; ++j) {
                OptTable[pr][j] = OptTable[pr][j] / pe;
            }
            for (i = 1; i <= row; ++i) {
                float pcoff = OptTable[i][pc];
                for (j = 1; j <= col; ++j) {
                    if (i != pr) {
                        OptTable[i][j] = OptTable[i][j] - (OptTable[pr][j]) * pcoff;
                    }
                }
            }
        }
    }

successed:
    for (i = 0; i <= row; ++i) {
        for (j = 0; j <= b; ++j) {
            if (OptTable[i][j] == 1) {
                printf("\tFor X%d == %0.3f , ", j, OptTable[i][col]);
            }
        }
    }
    printf("The optimum value is %0.3f\n ", OptTable[row][col]);
    return 0;
}
}
//----------------------------------

using namespace std;

//----------------------------------

namespace example1 {
#define MAX_N 1001
#define MAX_M 1001

typedef long long lld;
typedef unsigned long long llu;

int n, m;
double A[MAX_M][MAX_N], b[MAX_M], c[MAX_N], v;
int N[MAX_N], B[MAX_M]; // nonbasic & basic

// pivot yth variable around xth constraint
inline void pivot(int x, int y)
{
    printf("Pivoting variable %d around constraint %d.\n", y, x);
    
    // first rearrange the x-th row
    for (int j=0;j<n;j++)
    {
        if (j != y)
        {
            A[x][j] /= -A[x][y];
        }
    }
    b[x] /= -A[x][y];
    A[x][y] = 1.0 / A[x][y];
    
    // now rearrange the other rows
    for (int i=0;i<m;i++)
    {
        if (i != x)
        {
            for (int j=0;j<n;j++)
            {
                if (j != y)
                {
                    A[i][j] += A[i][y] * A[x][j];
                }
            }
            b[i] += A[i][y] * b[x];
            A[i][y] *= A[x][y];
        }
    }
    
    // now rearrange the objective function
    for (int j=0;j<n;j++)
    {
        if (j != y)
        {
            c[j] += c[y] * A[x][j];
        }
    }
    v += c[y] * b[x];
    c[y] *= A[x][y];
    
    // finally, swap the basic & nonbasic variable
    swap(B[x], N[y]);
}

// Run a single iteration of the simplex algorithm.
// Returns: 0 if OK, 1 if STOP, -1 if UNBOUNDED
inline int iterate_simplex()
{
    printf("--------------------\n");
    printf("State:\n");
    printf("Maximise: ");
    for (int j=0;j<n;j++) printf("%lfx_%d + ", c[j], N[j]);
    printf("%lf\n", v);
    printf("Subject to:\n");
    for (int i=0;i<m;i++)
    {
        for (int j=0;j<n;j++) printf("%lfx_%d + ", A[i][j], N[j]);
        printf("%lf = x_%d\n", b[i], B[i]);
    }
    
    // getchar(); // uncomment this for debugging purposes!
    
    int ind = -1, best_var = -1;
    for (int j=0;j<n;j++)
    {
        if (c[j] > 0)
        {
            if (best_var == -1 || N[j] < ind)
            {
                ind = N[j];
                best_var = j;
            }
        }
    }
    if (ind == -1) return 1;
    
    double max_constr = INFINITY;
    int best_constr = -1;
    for (int i=0;i<m;i++)
    {
        if (A[i][best_var] < 0)
        {
            double curr_constr = -b[i] / A[i][best_var];
            if (curr_constr < max_constr)
            {
                max_constr = curr_constr;
                best_constr = i;
            }
        }
    }
    if (isinf(max_constr)) return -1;
    else pivot(best_constr, best_var);
    
    return 0;
}

// (Possibly) converts the LP into a slack form with a feasible basic solution.
// Returns 0 if OK, -1 if INFEASIBLE
inline int initialise_simplex()
{
    int k = -1;
    double min_b = -1;
    for (int i=0;i<m;i++)
    {
        if (k == -1 || b[i] < min_b)
        {
            k = i;
            min_b = b[i];
        }
    }
    
    if (b[k] >= 0) // basic solution feasible!
    {
        for (int j=0;j<n;j++) N[j] = j;
        for (int i=0;i<m;i++) B[i] = n + i;
        return 0;
    }
    
    // generate auxiliary LP
    n++;
    for (int j=0;j<n;j++) N[j] = j;
    for (int i=0;i<m;i++) B[i] = n + i;
    
    // store the objective function
    double c_old[MAX_N];
    for (int j=0;j<n-1;j++) c_old[j] = c[j];
    double v_old = v;
    
    // aux. objective function
    c[n-1] = -1;
    for (int j=0;j<n-1;j++) c[j] = 0;
    v = 0;
    // aux. coefficients
    for (int i=0;i<m;i++) A[i][n-1] = 1;
    
    // perform initial pivot
    pivot(k, n - 1);
    
    // now solve aux. LP
    int code;
    while (!(code = iterate_simplex()));
    
    assert(code == 1); // aux. LP cannot be unbounded!!!
    
    if (v != 0) return -1; // infeasible!
    
    int z_basic = -1;
    for (int i=0;i<m;i++)
    {
        if (B[i] == n - 1)
        {
            z_basic = i;
            break;
        }
    }
    
    // if x_n basic, perform one degenerate pivot to make it nonbasic
    if (z_basic != -1) pivot(z_basic, n - 1);
    
    int z_nonbasic = -1;
    for (int j=0;j<n;j++)
    {
        if (N[j] == n - 1)
        {
            z_nonbasic = j;
            break;
        }
    }
    assert(z_nonbasic != -1);
    
    for (int i=0;i<m;i++)
    {
        A[i][z_nonbasic] = A[i][n-1];
    }
    swap(N[z_nonbasic], N[n - 1]);
    
    n--;
    for (int j=0;j<n;j++) if (N[j] > n) N[j]--;
    for (int i=0;i<m;i++) if (B[i] > n) B[i]--;
    
    for (int j=0;j<n;j++) c[j] = 0;
    v = v_old;
    
    for (int j=0;j<n;j++)
    {
        bool ok = false;
        for (int jj=0;jj<n;jj++)
        {
            if (j == N[jj])
            {
                c[jj] += c_old[j];
                ok = true;
                break;
            }
        }
        if (ok) continue;
        for (int i=0;i<m;i++)
        {
            if (j == B[i])
            {
                for (int jj=0;jj<n;jj++)
                {
                    c[jj] += c_old[j] * A[i][jj];
                }
                v += c_old[j] * b[i];
                break;
            }
        }
    }
    
    return 0;
}

// Runs the simplex algorithm to optimise the LP.
// Returns a vector of -1s if unbounded, -2s if infeasible.
pair<vector<double>, double> simplex()
{
    if (initialise_simplex() == -1)
    {
        return make_pair(vector<double>(n + m, -2), INFINITY);
    }
    
    int code;
    while (!(code = iterate_simplex()));
    
    if (code == -1) return make_pair(vector<double>(n + m, -1), INFINITY);
    
    vector<double> ret;
    ret.resize(n + m);
    for (int j=0;j<n;j++)
    {
        ret[N[j]] = 0;
    }
    for (int i=0;i<m;i++)
    {
        ret[B[i]] = b[i];
    }
    
    return make_pair(ret, v);
}

int main_example1()
{
    /*
     Simplex tests:
     
     Basic solution feasible:
     n = 2, m = 2;
     A[0][0] = -1; A[0][1] = 1;
     A[1][0] = -2; A[1][1] = -1;
     b[0] = 1; b[1] = 2;
     c[0] = 5; c[1] = -3;
     
     Basic solution feasible:
     n = 3, m = 3;
     A[0][0] = -1; A[0][1] = -1; A[0][2] = -3;
     A[1][0] = -2; A[1][1] = -2; A[1][2] = -5;
     A[2][0] = -4; A[2][1] = -1; A[2][2] = -2;
     b[0] = 30; b[1] = 24; b[2] = 36;
     c[0] = 3; c[1] = 1; c[2] = 2;
     
     Basic solution infeasible:
     n = 2, m = 3;
     A[0][0] = -1; A[0][1] = 1;
     A[1][0] = 1; A[1][1] = 1;
     A[2][0] = 1; A[2][1] = -4;
     b[0] = 8; b[1] = -3; b[2] = 2;
     c[0] = 1; c[1] = 3;
     
     LP infeasible:
     n = 2, m = 2;
     A[0][0] = -1; A[0][1] = -1;
     A[1][0] = 2; A[1][1] = 2;
     b[0] = 2; b[1] = -10;
     c[0] = 3; c[1] = -2;
     
     LP unbounded:
     n = 2, m = 2;
     A[0][0] = 2; A[0][1] = -1;
     A[1][0] = 1; A[1][1] = 2;
     b[0] = -1; b[1] = -2;
     c[0] = 1; c[1] = -1;
    */
    
    n = 2, m = 3;
    A[0][0] = -1; A[0][1] = 1;
    A[1][0] = 1; A[1][1] = 1;
    A[2][0] = 1; A[2][1] = -4;
    b[0] = 8; b[1] = -3; b[2] = 2;
    c[0] = 1; c[1] = 3;
    
    pair<vector<double>, double> ret = simplex();
    
    if (isinf(ret.second))
    {
        if (ret.first[0] == -1) printf("Objective function unbounded!\n");
        else if (ret.first[0] == -2) printf("Linear program infeasible!\n");
    }
    else
    {
        printf("Solution: (");
        for (int i=0;i<n+m;i++) printf("%lf%s", ret.first[i], (i < n + m - 1) ? ", " : ")\n");
        printf("Optimal objective value: %lf\n", ret.second);
    }
    
    return 0;
}
}
//----------------------------------

namespace example2 {
class Simplex
{

private:
    int rows, cols;
    // Stores coefficients of all the variables
    vector<vector<float>> A;
    // Stores constants of constraints
    vector<float> B;
    // Stores the coefficients of the objective function
    vector<float> C;

    float maximum;

    bool isUnbounded;

public:
    Simplex(vector<vector<float>> matrix, vector<float> b, vector<float> c)
    {
        maximum = 0;
        isUnbounded = false;
        rows = matrix.size();
        cols = matrix[0].size();
        A.resize(rows, vector<float>(cols, 0));
        B.resize(b.size());
        C.resize(c.size());

        for (int i = 0; i < rows; i++)
        { // Pass A[][] values to the matrix
            for (int j = 0; j < cols; j++)
            {
                A[i][j] = matrix[i][j];
            }
        }

        for (int i = 0; i < c.size(); i++)
        { // Pass c[] values to the C vector
            C[i] = c[i];
        }
        for (int i = 0; i < b.size(); i++)
        { // Pass b[] values to the B vector
            B[i] = b[i];
        }
    }

    bool simplexAlgorithmCalculataion()
    {
        // Check whether the table is optimal, if optimal no need to process further
        if (checkOptimality() == true)
        {
            return true;
        }

        // Find the column which has the pivot. The least coefficient of the objective function(C array).
        int pivotColumn = findPivotColumn();

        if (isUnbounded == true)
        {
            cout << "Error unbounded" << endl;
            return true;
        }

        // Find the row with the pivot value. The least value item's row in the B array
        int pivotRow = findPivotRow(pivotColumn);

        // Form the next table according to the pivot value
        doPivotting(pivotRow, pivotColumn);

        return false;
    }

    bool checkOptimality()
    {
        // If the table has further negative constraints,then it is not optimal
        bool isOptimal = false;
        int positveValueCount = 0;

        // Check if the coefficients of the objective function are negative
        for (int i = 0; i < C.size(); i++)
        {
            float value = C[i];
            if (value >= 0)
            {
                positveValueCount++;
            }
        }
        // If all the constraints are positive now,the table is optimal
        if (positveValueCount == C.size())
        {
            isOptimal = true;
            print();
        }
        return isOptimal;
    }

    void doPivotting(int pivotRow, int pivotColumn)
    {

        float pivetValue = A[pivotRow][pivotColumn]; // Gets the pivot value

        float pivotRowVals[cols]; // The column with the pivot

        float pivotColVals[rows]; // The row with the pivot

        float rowNew[cols]; // The row after processing the pivot value

        maximum = maximum - (C[pivotColumn] * (B[pivotRow] / pivetValue)); // Set the maximum step by step
        // Get the row that has the pivot value
        for (int i = 0; i < cols; i++)
        {
            pivotRowVals[i] = A[pivotRow][i];
        }
        // Get the column that has the pivot value
        for (int j = 0; j < rows; j++)
        {
            pivotColVals[j] = A[j][pivotColumn];
        }

        // Set the row values that has the pivot value divided by the pivot value and put into new row
        for (int k = 0; k < cols; k++)
        {
            rowNew[k] = pivotRowVals[k] / pivetValue;
        }

        B[pivotRow] = B[pivotRow] / pivetValue;

        // Process the other coefficients in the A array by subtracting
        for (int m = 0; m < rows; m++)
        {
            // Ignore the pivot row as we already calculated that
            if (m != pivotRow)
            {
                for (int p = 0; p < cols; p++)
                {
                    float multiplyValue = pivotColVals[m];
                    A[m][p] = A[m][p] - (multiplyValue * rowNew[p]);
                    // C[p] = C[p] - (multiplyValue*C[pivotRow]);
                    // B[i] = B[i] - (multiplyValue*B[pivotRow]);
                }
            }
        }

        // Process the values of the B array
        for (int i = 0; i < B.size(); i++)
        {
            if (i != pivotRow)
            {

                float multiplyValue = pivotColVals[i];
                B[i] = B[i] - (multiplyValue * B[pivotRow]);
            }
        }
        // The least coefficient of the constraints of the objective function
        float multiplyValue = C[pivotColumn];
        // Process the C array
        for (int i = 0; i < C.size(); i++)
        {
            C[i] = C[i] - (multiplyValue * rowNew[i]);
        }

        // Replacing the pivot row in the new calculated A array
        for (int i = 0; i < cols; i++)
        {
            A[pivotRow][i] = rowNew[i];
        }
    }

    // Print the current A array
    void print()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                cout << A[i][j] << " ";
            }
            cout << "" << endl;
        }
        cout << "" << endl;
    }

    // Find the least coefficients of constraints in the objective function's position
    int findPivotColumn()
    {

        int location = 0;
        float minm = C[0];

        for (int i = 1; i < C.size(); i++)
        {
            if (C[i] < minm)
            {
                minm = C[i];
                location = i;
            }
        }

        return location;
    }

    // Find the row with the pivot value.The least value item's row in the B array
    int findPivotRow(int pivotColumn)
    {
        float positiveValues[rows];
        vector<float> result(rows, 0);
        // Float result[rows];
        int negativeValueCount = 0;

        for (int i = 0; i < rows; i++)
        {
            if (A[i][pivotColumn] > 0)
            {
                positiveValues[i] = A[i][pivotColumn];
            }
            else
            {
                positiveValues[i] = 0;
                negativeValueCount += 1;
            }
        }
        // Checking the unbound condition if all the values are negative ones
        if (negativeValueCount == rows)
        {
            isUnbounded = true;
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                float value = positiveValues[i];
                if (value > 0)
                {
                    result[i] = B[i] / value;
                }
                else
                {
                    result[i] = 0;
                }
            }
        }
        // find the minimum's location of the smallest item of the B array
        float minimum = 99999999;
        int location = 0;
        for (int i = 0; i < sizeof(result) / sizeof(result[0]); i++)
        {
            if (result[i] > 0)
            {
                if (result[i] < minimum)
                {
                    minimum = result[i];

                    location = i;
                }
            }
        }

        return location;
    }

    void CalculateSimplex()
    {
        bool end = false;

        cout << "initial array(Not optimal)" << endl;
        print();

        cout << " " << endl;
        cout << "final array(Optimal solution)" << endl;

        while (!end)
        {

            bool result = simplexAlgorithmCalculataion();

            if (result == true)
            {

                end = true;
            }
        }
        cout << "Answers for the Constraints of variables" << endl;

        for (int i = 0; i < A.size(); i++)
        { // every basic column has the values, get it form B array
            int count0 = 0;
            int index = 0;
            for (int j = 0; j < rows; j++)
            {
                if (A[j][i] == 0.0)
                {
                    count0 += 1;
                }
                else if (A[j][i] == 1)
                {
                    index = j;
                }
            }

            if (count0 == rows - 1)
            {

                cout << "variable" << index + 1 << ": " << B[index] << endl; // every basic column has the values, get it form B array
            }
            else
            {
                cout << "variable" << index + 1 << ": " << 0 << endl;
            }
        }

        cout << "" << endl;
        cout << "maximum value: " << maximum << endl; // print the maximum values
    }
};

int main_example2()
{

    const int colSizeA = 6; // should initialise columns size in A
    const int rowSizeA = 3; // should initialise columns row in A[][] vector

    float C[] = {-6, -5, -4, 0, 0, 0}; // should initialis the c arry here
    float B[] = {180, 300, 240};       // should initialis the b array here

    float a[rowSizeA][colSizeA] = {// should intialis the A[][] array here
                     {2, 1, 1, 1, 0, 0},
                     {1, 3, 2, 0, 1, 0},
                     {2, 1, 2, 0, 0, 1}};

    vector<vector<float>> vec2D(rowSizeA, vector<float>(colSizeA, 0));

    vector<float> b(rowSizeA, 0);
    vector<float> c(colSizeA, 0);

    for (int i = 0; i < rowSizeA; i++)
    { // make a vector from given array
        for (int j = 0; j < colSizeA; j++)
        {
            vec2D[i][j] = a[i][j];
        }
    }

    for (int i = 0; i < rowSizeA; i++)
    {
        b[i] = B[i];
    }

    for (int i = 0; i < colSizeA; i++)
    {
        c[i] = C[i];
    }

    // hear the make the class parameters with A[m][n] vector b[] vector and c[] vector
    Simplex simplex(vec2D, b, c);
    simplex.CalculateSimplex();

    return 0;
}
}

//----------------------------------
//----------------------------------
//----------------------------------

using namespace example0;
using namespace example1;
using namespace example2;

int main()
{
	cout << endl << "******************************" << endl;
	cout << "******************************" << endl;
	main_example0();
	cout << endl << "******************************" << endl;
	cout << "******************************" << endl << endl;
	main_example1();
	cout << endl << "******************************" << endl;
	cout << "******************************" << endl << endl;
	main_example2();
	cout << endl << "******************************" << endl;
	cout << "******************************" << endl << endl;
	return 0;
}
//----------------------------------
//----------------------------------
//----------------------------------
