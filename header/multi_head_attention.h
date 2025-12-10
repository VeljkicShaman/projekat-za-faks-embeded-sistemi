#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <systemc.h>
#include <vector>
#include <iomanip>
#include "datatypes.h"
#include "single_head_attention.h"

SC_MODULE(MultiHeadAttentionModule) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;
    
    const Matrix* Q_in_ptr = nullptr;
    const Matrix* K_in_ptr = nullptr;
    const Matrix* V_in_ptr = nullptr;
    const Matrix* W_out_ptr = nullptr;
    const Vector* b_out_ptr = nullptr;
    Matrix* Y_out_ptr = nullptr;

    int num_heads;
    sc_vector<SingleHeadAttentionModule> attention_heads;
    MatrixMultiplier final_proj_unit;
    
    sc_vector<sc_signal<bool>> head_starts;
    sc_vector<sc_signal<bool>> head_dones;
    sc_signal<bool> final_proj_start, final_proj_done;

    Matrix3D q_heads_data, k_heads_data, v_heads_data, attn_output_heads_data;
    Matrix merged_heads_output;

    void multi_head_process() {
        done.write(false);
        while(true) {
            wait(start.posedge_event());
            
            size_t seq_len = Q_in_ptr->size();
            size_t embed_dim = Q_in_ptr->at(0).size();
            size_t head_dim = embed_dim / num_heads;

            q_heads_data.resize(num_heads, Matrix(seq_len, Vector(head_dim)));
            k_heads_data.resize(num_heads, Matrix(seq_len, Vector(head_dim)));
            v_heads_data.resize(num_heads, Matrix(seq_len, Vector(head_dim)));
            attn_output_heads_data.resize(num_heads, Matrix(seq_len, Vector(head_dim)));
            merged_heads_output.resize(seq_len, Vector(embed_dim));

            //Priprema podataka
            for (int h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < head_dim; ++j) {
                        q_heads_data[h][i][j] = Q_in_ptr->at(i)[h * head_dim + j];
                        k_heads_data[h][i][j] = K_in_ptr->at(i)[h * head_dim + j];
                        v_heads_data[h][i][j] = V_in_ptr->at(i)[h * head_dim + j];
                    }
                }
                attention_heads[h].Q_ptr = &q_heads_data[h];
                attention_heads[h].K_ptr = &k_heads_data[h];
                attention_heads[h].V_ptr = &v_heads_data[h];
                attention_heads[h].Y_ptr = &attn_output_heads_data[h];
            }

            std::cout << "@" << sc_time_stamp() << "Obrada glava (trenutno radi sekvencijalno, probacu paralelno kada spustim na plocu)..." << std::endl;

            //Izvrsavanje glava
            for (int h = 0; h < num_heads; ++h) {
                head_starts[h].write(true);
                wait(head_dones[h].posedge_event());
                head_starts[h].write(false);
                wait(clk->posedge_event());
                std::cout << "@" << sc_time_stamp() << "Zavrsio glavu " << h << std::endl;
            }
            
            //Merge
            for (int h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < head_dim; ++j) {
                        merged_heads_output[i][h * head_dim + j] = attn_output_heads_data[h][i][j];
                    }
                }
            }
            
            //Final Projection
            final_proj_unit.X_ptr = &merged_heads_output;
            final_proj_unit.W_ptr = W_out_ptr; 
            final_proj_unit.b_ptr = b_out_ptr;
            final_proj_unit.Y_ptr = Y_out_ptr;
            
            final_proj_start.write(true);
            wait(final_proj_done.posedge_event());
            final_proj_start.write(false);
            
            wait(clk->posedge_event());
            done.write(true);
            wait(clk->posedge_event());
            done.write(false);
        }
    }

    SC_CTOR(MultiHeadAttentionModule) : 
        attention_heads("heads", 8), final_proj_unit("FinalProjectionUnit"),
        head_starts("starts", 8), head_dones("dones", 8)
    {
        num_heads = 8;
        for(int i=0; i<num_heads; ++i) {
            attention_heads[i].clk(clk);
            attention_heads[i].start(head_starts[i]);
            attention_heads[i].done(head_dones[i]);
        }
        SC_THREAD(multi_head_process);
        sensitive << clk.pos();
        final_proj_unit.clk(clk);
        final_proj_unit.start(final_proj_start);
        final_proj_unit.done(final_proj_done);
    }
};
#endif