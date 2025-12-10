#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


using Matrix = std::vector<std::vector<float>>;
using Vector = std::vector<float>;
using Matrix3D = std::vector<Matrix>;
namespace py = pybind11;


Matrix matmul_standard(const Matrix& A, const Matrix& B);
Matrix matmul_transpose(const Matrix& A, const Matrix& B, const Vector& bias = {});
Matrix softmax_internal(const Matrix& mat);

//Glavna funkcija
Matrix multi_head_attention_core(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads) {
    size_t seq_len = Q.size();
    size_t embed_dim = Q[0].size();
    size_t head_dim = embed_dim / num_heads;

    Matrix3D q_heads(num_heads, Matrix(seq_len, Vector(head_dim)));
    Matrix3D k_heads(num_heads, Matrix(seq_len, Vector(head_dim)));
    Matrix3D v_heads(num_heads, Matrix(seq_len, Vector(head_dim)));

    for (int h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                q_heads[h][i][j] = Q[i][h * head_dim + j];
                k_heads[h][i][j] = K[i][h * head_dim + j];
                v_heads[h][i][j] = V[i][h * head_dim + j];
            }
        }
    }
    
    Matrix3D attn_output_heads(num_heads, Matrix(seq_len, Vector(head_dim)));
    //float scale_factor = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
        Matrix scores = matmul_transpose(q_heads[h], k_heads[h]);
       // for (auto& row : scores) { for (auto& val : row) { val *= scale_factor; } }
        Matrix probs = softmax_internal(scores);
        attn_output_heads[h] = matmul_standard(probs, v_heads[h]);
    }

    Matrix merged_output(seq_len, Vector(embed_dim));
    for (int h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                merged_output[i][h * head_dim + j] = attn_output_heads[h][i][j];
            }
        }
    }
    return merged_output;
}


PYBIND11_MODULE(multihead_attention_algorithm, m) {
    m.doc() = "C++ modul za Multi-Head Attention";
    m.def("attention_core", &multi_head_attention_core, "MHA bez final proj",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("num_heads")
    );
}

//Implementacije pomoćnih funkcija
Matrix matmul_standard(const Matrix& A, const Matrix& B) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();
	
    if (A_cols != B_rows) { throw std::runtime_error("Dimenzije za A * B se ne poklapaju!"); }
    Matrix C(A_rows, Vector(B_cols, 0.0f));
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix matmul_transpose(const Matrix& A, const Matrix& B, const Vector& bias) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
	
    if (A_cols != B[0].size()) {
		throw std::runtime_error("Dimenzije za A * B^T se ne poklapaju!"); 
	}
    Matrix C(A_rows, Vector(B_rows, 0.0f));
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_rows; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A_cols; ++k) {
                sum += A[i][k] * B[j][k];
            }
            if (!bias.empty()) { C[i][j] = sum + bias[j]; }
            else { C[i][j] = sum; }
        }
    }
    return C;
}

Matrix softmax_internal(const Matrix& mat) {
    Matrix result = mat;
    for (size_t i = 0; i < mat.size(); ++i) {
        float max_val = mat[i][0];
        for (size_t j = 1; j < mat[i].size(); ++j) {
			if (mat[i][j] > max_val) max_val = mat[i][j]; 
		}
        float sum_exp = 0.0f;
        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[i][j] = expf(mat[i][j] - max_val);
            sum_exp += result[i][j];
        }
        for (size_t j = 0; j < mat[i].size(); ++j) { 
			result[i][j] /= sum_exp; 
		}
    }
    return result;
}