#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <iomanip>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using Matrix3D = std::vector<Matrix>;

//Deklaracije 
Matrix readMatrix(const std::string& filename);
Vector readVector(const std::string& filename);
void writeMatrix(const std::string& filename, const Matrix& mat);

void analyze_bits(const std::string& name, const Matrix& mat) {
    double max_val = -1e9;
    double min_abs_non_zero = 1e9;

    for (const auto& row : mat) {
        for (double val : row) {
            double abs_val = std::abs(val);
            if (val > max_val) max_val = val;
            
            // Trazimo najmanju vrednost vecu od nule
            if (abs_val > 0.0000000001 && abs_val < min_abs_non_zero) {
                min_abs_non_zero = abs_val;
            }
        }
    }

    int int_bits = 0;
    double abs_max = std::max(std::abs(max_val), std::abs(min_abs_non_zero)); 
    
    double true_abs_max = 0;
    for (const auto& row : mat) for (double v : row) if(std::abs(v) > true_abs_max) true_abs_max = std::abs(v);
    
    if (true_abs_max != 0) {
        int_bits = static_cast<int>(std::ceil(std::log2(true_abs_max)));
        if (int_bits < 1) int_bits = 1; // Uvek bar 1 bit za celi deo (0 ili 1)
    }
    int_bits += 1; // +1 za znak

    //2. Fract bits
    //formula: log2(2^-N)
    int frac_bits = 0;
    if (min_abs_non_zero < 1e9) { //ako smo nasli neki mali broj
        frac_bits = static_cast<int>(std::ceil(std::log2(1.0 / min_abs_non_zero)));
    }
    
    //ogranicavanje na 32 bita
    if (frac_bits > 20) frac_bits = 20;//limit
    if (frac_bits < 0) frac_bits = 0;

    int total_bits = int_bits + frac_bits;

    std::cout << "Analiza: " << std::left << std::setw(25) << name 
              << " | Opseg: (" << min_abs_non_zero << " ... " << true_abs_max << ")"
              << " | Potreban format: Q" << int_bits << "." << frac_bits 
              << " (Ukupno: " << total_bits << " bita)" << std::endl;
}

//Mnozenje Red x Red za Q*K
Matrix matmul_transpose(const Matrix& A, const Matrix& B) {
    size_t rows = A.size(); size_t cols = A[0].size(); size_t B_rows = B.size();
    Matrix C(rows, Vector(B_rows, 0.0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < B_rows; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) sum += A[i][k] * B[j][k];
            C[i][j] = sum;
        }
    }
    return C;
}

//Mnozenje matrica Red x Kolona
Matrix matmul_standard(const Matrix& A, const Matrix& B) {
    size_t A_rows = A.size(); size_t A_cols = A[0].size(); 
    size_t B_rows = B.size(); size_t B_cols = B[0].size();
    Matrix C(A_rows, Vector(B_cols, 0.0));
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A_cols; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    return C;
}

Matrix softmax_internal(const Matrix& mat) {
    Matrix result = mat;
    for (size_t i = 0; i < mat.size(); ++i) {
        double max_val = mat[i][0];
        for (size_t j = 1; j < mat[i].size(); ++j) if (mat[i][j] > max_val) max_val = mat[i][j];
        double sum_exp = 0.0;
        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[i][j] = std::exp(mat[i][j] - max_val);
            sum_exp += result[i][j];
        }
        for (size_t j = 0; j < mat[i].size(); ++j) result[i][j] /= sum_exp;
    }
    return result;
}

Matrix multi_head_attention_realtime(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads) {
    size_t seq_len = Q.size(); size_t embed_dim = Q[0].size(); size_t head_dim = embed_dim / num_heads;
	std::cout << " - POCETAK BITSKE ANALIZE - " << std::endl;
	analyze_bits("Ulaz Q ", Q);
	analyze_bits("Ulaz K ", K);
	analyze_bits("Ulaz V ", V);
    Matrix3D q_heads(num_heads, Matrix(seq_len, Vector(head_dim)));
    Matrix3D k_heads(num_heads, Matrix(seq_len, Vector(head_dim)));
    Matrix3D v_heads(num_heads, Matrix(seq_len, Vector(head_dim)));

    //Splitujemo glave
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
	
	//skaliranje sam izbacio jer to sada radi python
    for (int h = 0; h < num_heads; ++h) {
        //1. Mnozenje
        Matrix scores = matmul_transpose(q_heads[h], k_heads[h]);
        analyze_bits("Head " + std::to_string(h) + " Scores", scores);
        //2. Softmax (bez skaliranja)
        Matrix probs = softmax_internal(scores);
        analyze_bits("Head " + std::to_string(h) + " Scores(Softmax)", probs);
        //3. Output
        Matrix V_curr(seq_len, Vector(head_dim));
        for(size_t i=0; i<seq_len; ++i)
            for(size_t j=0; j<head_dim; ++j)
                V_curr[i][j] = v_heads[h][i][j];

        attn_output_heads[h] = matmul_standard(probs, V_curr);
		analyze_bits("Head " + std::to_string(h) + " Output", attn_output_heads[h]);
    }

    // Merge
    Matrix merged_output(seq_len, Vector(embed_dim)) ;
    for (int h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                merged_output[i][h * head_dim + j] = attn_output_heads[h][i][j];
            }
        }
    }
	analyze_bits("Merged Output ", merged_output);
    return merged_output;
}

int main() {
    std::cout << "Pokrecemo C++ referencu..." << std::endl;
    Matrix Q = readMatrix("matrice/multihead_ulaz_Q.txt");
    Matrix K = readMatrix("matrice/multihead_ulaz_K.txt");
    Matrix V = readMatrix("matrice/multihead_ulaz_V.txt");
    Matrix W = readMatrix("matrice/multihead_W_out.txt");
    Vector b = readVector("matrice/multihead_b_out.txt");
	   
    if (Q.empty()) return 1;

    Matrix merged = multi_head_attention_realtime(Q, K, V, 8);
    
    //Finalna projekcija
    Matrix final = matmul_standard(merged, W);
    for (size_t i = 0; i < final.size(); ++i) {
        for (size_t j = 0; j < final[0].size(); ++j) { 
            final[i][j] += b[j];
		}
	}
	analyze_bits("Final Output ", final);
    writeMatrix("izlaz_cpp.txt", final);
    std::cout << "Izlazni fajl je kreiran." << std::endl;
	return 0;
}

//I/O
Matrix readMatrix(const std::string& filename) {
    Matrix mat; std::ifstream file(filename);
    if (!file.is_open()) return mat;
    std::string line;
    while (std::getline(file, line)) {
        Vector row; std::stringstream ss(line); double val;
        while (ss >> val) { row.push_back(val); }
        if (!row.empty()) { mat.push_back(row); }
    }
    return mat;
}
Vector readVector(const std::string& filename) {
    Vector vec; std::ifstream file(filename);
    if (!file.is_open()) return vec;
    double val; while (file >> val) { vec.push_back(val); }
    return vec;
}
void writeMatrix(const std::string& filename, const Matrix& mat) {
    std::ofstream file(filename);
    for (const auto& row : mat) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i] << (i == row.size() - 1 ? "" : " ");
        }
        file << std::endl;
    }
}