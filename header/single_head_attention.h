#ifndef SINGLE_HEAD_ATTENTION_H
#define SINGLE_HEAD_ATTENTION_H

#include <systemc.h>
#include <cmath>
#include <vector>
#include "datatypes.h"
#include "matrix_multiplier.h"


inline std::vector<std::vector<PROB_T>> softmax_safe(const Matrix& mat) {
    std::vector<std::vector<PROB_T>> result(mat.size(), std::vector<PROB_T>(mat[0].size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        double max_val = mat[i][0].to_double();
        for (size_t j = 1; j < mat[i].size(); ++j) { 
            if (mat[i][j].to_double() > max_val) max_val = mat[i][j].to_double(); 
        }
        double sum_exp = 0.0;
        std::vector<double> exp_vals(mat[i].size());
        for (size_t j = 0; j < mat[i].size(); ++j) {
            exp_vals[j] = std::exp(mat[i][j].to_double() - max_val);
            sum_exp += exp_vals[j];
        }
        for (size_t j = 0; j < mat[i].size(); ++j) { 
            result[i][j] = exp_vals[j] / sum_exp; 
        }
    }
    return result;
}

SC_MODULE(SingleHeadAttentionModule) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;
    
    const Matrix* Q_ptr = nullptr;
    const Matrix* K_ptr = nullptr;
    const Matrix* V_ptr = nullptr;
    Matrix* Y_ptr = nullptr;
    
    MatrixMultiplier mat_mul_unit;
    sc_signal<bool> mat_mul_start_sig, mat_mul_done_sig;
    
    Matrix scores;
    Matrix probs; 

    void attention_process() {
        done.write(false);
        while (true) {
            wait(start.posedge_event());

            //1. Q * K^T
            scores.assign(Q_ptr->size(), Vector(K_ptr->size()));
            mat_mul_unit.X_ptr = Q_ptr; mat_mul_unit.W_ptr = K_ptr; mat_mul_unit.b_ptr = nullptr; mat_mul_unit.Y_ptr = &scores;
            
            mat_mul_start_sig.write(true);
            wait(mat_mul_done_sig.posedge_event());
            mat_mul_start_sig.write(false);
            wait(clk->posedge_event());
            
            //ovde sam inace radio skaliranje, ali sada to radi softver
            
            //2. Softmax
            std::vector<std::vector<PROB_T>> probs_precise = softmax_safe(scores);
            probs.assign(probs_precise.size(), Vector(probs_precise[0].size()));
            
            for(size_t i=0; i<probs_precise.size(); ++i) {
                for(size_t j=0; j<probs_precise[0].size(); ++j) {
                    double p_val = probs_precise[i][j];
                    if (p_val >= 1.0) p_val = 0.999;
                    probs[i][j] = p_val; 
                }
            }

            //3. Probs * V
            Matrix V_T(V_ptr->at(0).size(), Vector(V_ptr->size()));
            for(size_t i=0; i < V_ptr->size(); ++i) {
                for(size_t j=0; j < V_ptr->at(0).size(); ++j) {
                    V_T[j][i] = V_ptr->at(i)[j]; 
                } 
            }
            
            mat_mul_unit.X_ptr = &probs; mat_mul_unit.W_ptr = &V_T; mat_mul_unit.b_ptr = nullptr; mat_mul_unit.Y_ptr = Y_ptr;
            
            mat_mul_start_sig.write(true);
            wait(mat_mul_done_sig.posedge_event());
            mat_mul_start_sig.write(false);
            
            done.write(true);
            wait(clk->posedge_event());
            done.write(false);
        }
    }

    SC_CTOR(SingleHeadAttentionModule) : mat_mul_unit("MatMultiplier") {
        SC_THREAD(attention_process);
        sensitive << clk.pos();
        mat_mul_unit.clk(clk);
        mat_mul_unit.start(mat_mul_start_sig);
        mat_mul_unit.done(mat_mul_done_sig);
    }
};
#endif