#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

#include <systemc.h>
#include <iostream> 
#include "datatypes.h"

SC_MODULE(MatrixMultiplier) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    const Matrix* X_ptr = nullptr;
    const Matrix* W_ptr = nullptr;
    const Vector* b_ptr = nullptr;
    Matrix* Y_ptr = nullptr;

    void multiply_process() {
        done.write(false);
        while (true) {
            //cekamo start signal
            do { wait(clk->posedge_event()); } while (start.read() == false);

            const Matrix& X = *X_ptr;
            const Matrix& W = *W_ptr;
            Matrix& Y = *Y_ptr;
            
            size_t seq_len = X.size();
            size_t in_feat = X[0].size();
            size_t out_feat = W.size();
            
            //logika
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < out_feat; ++j) {
                    ACC_T sum = 0.0; //32-bitni akumulator
                    
                    //Najzahtevniji deo (MAC operacije)
                    for (size_t k = 0; k < in_feat; ++k) {
                        sum += X[i][k] * W[j][k];
                    }
                    
                    //Upis rezultata (sa ili bez biasa)
                    if (b_ptr) { 
                        Y[i][j] = sum + b_ptr->at(j); 
                    } else { 
                        Y[i][j] = sum; 
                    }
                }
            }

            
            //signaliziramo kraj
            done.write(true);
            

            while (start.read() == true) { wait(clk->posedge_event()); }
            done.write(false);
        }
    }

    SC_CTOR(MatrixMultiplier) {
        SC_THREAD(multiply_process);
    }
};

#endif // MATRIX_MULTIPLIER_H