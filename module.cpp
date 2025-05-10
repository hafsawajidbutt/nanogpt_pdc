#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &b, int &x, int &y, int &z, 
    const int &sizeX, const int &sizeY, const int &sizeZ) {
return tensor[b * sizeX * sizeY * sizeZ + x * sizeY * sizeZ + y * sizeZ + z];
}

inline void fourDimWrite(std::vector<float> &tensor, int &b, int &x, int &y, int &z, 
    const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
tensor[b * sizeX * sizeY * sizeZ + x * sizeY * sizeZ + y * sizeZ + z] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //

    // loop over batches
    for (int b = 0; b < B; b++) 
    {
    // loop over heads
        for (int h = 0; h < H; h++) 
        {
    
            // Step 1: Q * K^T
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) 
                    {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
                        float k_val = fourDimRead(K, b, h, j, k, H, N, d); // no transpose needed
                        sum += q_val * k_val;
                    }
                    twoDimWrite(QK_t, i, j, N, sum);
                }
            }

            // Step 2: Softmax row-wise on QK_t
            for (int i = 0; i < N; i++) 
            {
                float row_sum = 0.0f;
                for (int j = 0; j < N; j++) 
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = std::exp(val);
                    twoDimWrite(QK_t, i, j, N, val); // overwrite with exp(val)
                    row_sum += val;
                }
                for (int j = 0; j < N; j++) 
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= row_sum;
                    twoDimWrite(QK_t, i, j, N, val); // overwrite with softmax val
                }
            }

            // Step 3: Output = softmax(QK^T) * V
            for (int i = 0; i < N; i++) 
            {
                for (int k = 0; k < d; k++) 
                {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) 
                    {
                        float attn_weight = twoDimRead(QK_t, i, j, N);
                        float v_val = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += attn_weight * v_val;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d)
{
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    // loop over batches
    for (int b = 0; b < B; b++) 
    {
        // loop over heads
        for (int h = 0; h < H; h++) 
        {
                
            // Step 1: Q * K^T
            for (int i = 0; i < N; i++) 
            {
                for (int j = 0; j < N; j++) 
                {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) 
                    {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
                        float k_val = fourDimRead(K, b, h, j, k, H, N, d); // no transpose needed
                        sum += q_val * k_val;
                    }
                    twoDimWrite(QK_t, i, j, N, sum);
                }
            }
    
            // Step 2: Softmax row-wise on QK_t
            for (int i = 0; i < N; i++) 
            {
                float row_sum = 0.0f;
                for (int j = 0; j < N; j++) 
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = std::exp(val);
                    twoDimWrite(QK_t, i, j, N, val); // overwrite with exp(val)
                    row_sum += val;
                }
                for (int j = 0; j < N; j++) 
                {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= row_sum;
                    twoDimWrite(QK_t, i, j, N, val); // overwrite with softmax val
                }
            }
    
            // Step 3: Output = softmax(QK^T) * V
            for (int i = 0; i < N; i++) 
            {
                for (int k = 0; k < d; k++) 
                {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) 
                    {
                        float attn_weight = twoDimRead(QK_t, i, j, N);
                        float v_val = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += attn_weight * v_val;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
    int B, int H, int N, int d)
{
    //code for fused attention
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    // Outer loops — parallelize across (b, h, i)
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) 
    {
        for (int h = 0; h < H; h++) 
        {
            for (int i = 0; i < N; i++) 
            {
                // Each thread gets its own row buffer
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);

                // Step 1: Q[i] ⋅ K^T (row i of QK^T)
                float row_sum = 0.0f;
                for (int j = 0; j < N; j++) 
                {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) 
                    {
                        float q = fourDimRead(Q, b, h, i, k, H, N, d);
                        float kk = fourDimRead(K, b, h, j, k, H, N, d);
                        dot += q * kk;
                    }
                    float softmax_val = std::exp(dot);
                    ORow[j] = softmax_val;
                    row_sum += softmax_val;
                }

                // Step 2: Normalize softmax
                for (int j = 0; j < N; j++) 
                {
                ORow[j] /= row_sum;
                }

                // Step 3: O[i] = softmax(row) × V
                for (int k = 0; k < d; k++) 
                {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) 
                    {
                        float p = ORow[j]; // softmax weight
                        float v = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += p * v;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                }
            }
        }
    }

    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}



// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            for (int i0 = 0; i0 < N; i0 += Br) {
                int BrEff = std::min(Br, N - i0);

                // Qi ← Q[b,h,i0:i0+Br]
                for (int i = 0; i < BrEff; i++) {
                    for (int k = 0; k < d; k++) {
                        float val = fourDimRead(Q, b, h, i0 + i, k, H, N, d);
                        twoDimWrite(Qi, i, k, d, val);
                        twoDimWrite(Oi, i, k, d, 0.0f);
                    }
                    li[i] = 0.0f;
                }

                for (int j0 = 0; j0 < N; j0 += Bc) {
                    int BcEff = std::min(Bc, N - j0);

                    // Kj ← K[b,h,j0:j0+Bc], Vj ← V[b,h,j0:j0+Bc]
                    for (int j = 0; j < BcEff; j++) {
                        for (int k = 0; k < d; k++) {
                            twoDimWrite(Kj, j, k, d, fourDimRead(K, b, h, j0 + j, k, H, N, d));
                            twoDimWrite(Vj, j, k, d, fourDimRead(V, b, h, j0 + j, k, H, N, d));
                        }
                    }

                    // Sij = Qi × Kjᵗ
                    for (int i = 0; i < BrEff; i++) {
                        for (int j = 0; j < BcEff; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < d; k++) {
                                float q = twoDimRead(Qi, i, k, d);
                                float k_val = twoDimRead(Kj, j, k, d);
                                sum += q * k_val;
                            }
                            twoDimWrite(Sij, i, j, Bc, sum);
                        }
                    }

                    // Pij = softmax(Sij)
                    for (int i = 0; i < BrEff; i++) {
                        float row_sum = 0.0f;
                        for (int j = 0; j < BcEff; j++) {
                            float val = std::exp(twoDimRead(Sij, i, j, Bc));
                            twoDimWrite(Pij, i, j, Bc, val);
                            row_sum += val;
                        }
                        lij[i] = row_sum;

                        for (int j = 0; j < BcEff; j++) {
                            float val = twoDimRead(Pij, i, j, Bc) / row_sum;
                            twoDimWrite(Pij, i, j, Bc, val);
                        }
                    }

                    // PV = Pij × Vj
                    for (int i = 0; i < BrEff; i++) {
                        for (int k = 0; k < d; k++) {
                            float sum = 0.0f;
                            for (int j = 0; j < BcEff; j++) {
                                float p = twoDimRead(Pij, i, j, Bc);
                                float v = twoDimRead(Vj, j, k, d);
                                sum += p * v;
                            }
                            twoDimWrite(PV, i, k, d, sum);
                        }
                    }

                    // Oi += PV (accumulate)
                    for (int i = 0; i < BrEff; i++) {
                        lnew[i] = li[i] + lij[i];
                        for (int k = 0; k < d; k++) {
                            float o_val = twoDimRead(Oi, i, k, d);
                            float pv = twoDimRead(PV, i, k, d);
                            float updated = (li[i] * o_val + lij[i] * pv) / lnew[i];
                            twoDimWrite(Oi, i, k, d, updated);
                        }
                        li[i] = lnew[i]; // update denominator
                    }
                }

                // Write Oi back into O
                for (int i = 0; i < BrEff; i++) {
                    for (int k = 0; k < d; k++) {
                        float val = twoDimRead(Oi, i, k, d);
                        fourDimWrite(O, b, h, i0 + i, k, H, N, d, val);
                    }
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
