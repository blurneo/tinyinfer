#pragma once

#include <iostream>
#include <memory>
#include "flow/vector.h"

namespace tf
{

    template <int row_size, int col_size>
    struct Matrix
    {
        std::vector<float> vals;
        Matrix() { vals.resize(row_size * col_size); }
        const float *operator[](int row) const
        {
            return &(vals[row_size * col_size]);
        }
        float *operator[](int row)
        {
            return &(vals[row * col_size]);
        }
        // inverse matrix from: https://www.geeksforgeeks.org/adjoint-inverse-matrix/
        // Function to get cofactor of A[p][q] in temp[][]. n is
        // current dimension of A[][]
        template <typename = std::enable_if_t<row_size == col_size>>
        void getCofactor(Matrix &temp, int p, int q, int n) const
        {
            int i = 0, j = 0;

            // Looping for each element of the matrix
            for (int row = 0; row < n; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    //  Copying into temporary matrix only those
                    //  element which are not in given row and
                    //  column
                    if (row != p && col != q)
                    {
                        temp[i][j++] = this->operator[](row)[col];

                        // Row is filled, so increase row index and
                        // reset col index
                        if (j == n - 1)
                        {
                            j = 0;
                            i++;
                        }
                    }
                }
            }
        }

        /* Recursive function for finding determinant of matrix.
        n is current dimension of A[][]. */
        template <typename = std::enable_if_t<row_size == col_size>>
        int determinant(int n) const
        {
            int D = 0; // Initialize result

            //  Base case : if matrix contains single element
            if (n == 1)
                return this->operator[](0)[0];

            Matrix<row_size, col_size> temp; // To store cofactors

            int sign = 1; // To store sign multiplier

            // Iterate for each element of first row
            for (int f = 0; f < n; f++)
            {
                // Getting Cofactor of A[0][f]
                getCofactor(temp, 0, f, n);
                D += sign * this->operator[](0)[f] * determinant(n - 1);

                // terms are to be added with alternate sign
                sign = -sign;
            }

            return D;
        }

        // Function to get adjoint of A[N][N] in adj[N][N].
        template <typename = std::enable_if_t<row_size == col_size>>
        void adjoint(Matrix<row_size, col_size> &adj) const
        {
            if (row_size == 1)
            {
                adj[0][0] = 1;
                return;
            }

            // temp is used to store cofactors of A[][]
            int sign = 1;
            Matrix<row_size, col_size> temp;

            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                {
                    // Get cofactor of A[i][j]
                    getCofactor(temp, i, j, row_size);

                    // sign of adj[j][i] positive if sum of row
                    // and column indexes is even.
                    sign = ((i + j) % 2 == 0) ? 1 : -1;

                    // Interchanging rows and columns to get the
                    // transpose of the cofactor matrix
                    adj[j][i] = (sign) * (temp.determinant(row_size - 1));
                }
            }
        }

        // Function to calculate and store inverse, returns false if
        // matrix is singular
        bool inverse(Matrix<row_size, col_size> &inverse) const
        {
            // Find determinant of A[][]
            int det = determinant(row_size);
            if (det == 0)
            {
                std::cout << "Singular matrix, can't find its inverse";
                return false;
            }

            // Find adjoint
            Matrix<row_size, col_size> adj;
            adjoint(adj);

            // Find Inverse using formula "inverse(A) =
            // adj(A)/det(A)"
            for (int i = 0; i < row_size; i++)
                for (int j = 0; j < col_size; j++)
                    inverse[i][j] = adj[i][j] / float(det);

            return true;
        }

        // Generic function to display the matrix.  We use it to
        // display both adjoin and inverse. adjoin is integer matrix
        // and inverse is a float.
        void display()
        {
            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                    std::cout << this->operator[](i)[j] << " ";
                std::cout << std::endl;
            }
        }
    };
    typedef Matrix<2, 2> Matrix2x2;
}
