#define SC_INCLUDE_FX
#include <systemc.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

#include "datatypes.h"
#include "multi_head_attention.h"

Matrix readMatrix(const std::string& filename);
Vector readVector(const std::string& filename);
void writeMatrix(const std::string& filename, const Matrix& mat);

Matrix transpose(const Matrix& mat) {
    if (mat.empty()) return mat;
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Matrix trans(cols, Vector(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            trans[j][i] = mat[i][j];
        }
    }
    return trans;
}

SC_MODULE(Testbench) {
    sc_clock clk;
    sc_signal<bool> start_sig, done_sig;
    MultiHeadAttentionModule uut;
    
    Matrix Q_data, K_data, V_data, W_out_data_raw, W_out_data_transposed, Y_data;
    Vector b_out_data;

    void stimulus_process() {
        std::cout << "Ucitavanje fajlova..." << std::endl;
        Q_data = readMatrix("matrice/multihead_ulaz_Q.txt");
        K_data = readMatrix("matrice/multihead_ulaz_K.txt");
        V_data = readMatrix("matrice/multihead_ulaz_V.txt");
        W_out_data_raw = readMatrix("matrice/multihead_W_out.txt");
        b_out_data = readVector("matrice/multihead_b_out.txt");
        
        if (Q_data.empty()) return;

        //Ovde transponujemo W zbog izlaza
        W_out_data_transposed = transpose(W_out_data_raw);

        Y_data.resize(Q_data.size(), Vector(Q_data[0].size()));

        uut.Q_in_ptr = &Q_data;
        uut.K_in_ptr = &K_data;
        uut.V_in_ptr = &V_data;
        uut.W_out_ptr = &W_out_data_transposed; 
        uut.b_out_ptr = &b_out_data;
        uut.Y_out_ptr = &Y_data;

        wait(10, SC_NS);
        start_sig.write(true);
        wait(clk.posedge_event());
        start_sig.write(false);

        wait(done_sig.posedge_event());
        
        writeMatrix("izlaz_multihead_systemc.txt", Y_data);
        std::cout << "Simulacija zavrsena." << std::endl;
        sc_stop();
    }

    SC_CTOR(Testbench) : clk("clk", 10, SC_NS), uut("MultiHead_UUT") {
        uut.clk(clk); uut.start(start_sig); uut.done(done_sig);
        SC_THREAD(stimulus_process);
    }
};

int sc_main(int argc, char* argv[]) {
    Testbench tb("Testbench_1");
    sc_start();
    return 0;
}

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