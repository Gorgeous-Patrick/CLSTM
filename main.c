#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data; // Flattened matrix
    int rows;
    int cols;
} Matrix;

void matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {
    // C = A * B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->rows, B->cols, A->cols, 1.0f,
                A->data, A->cols, B->data, B->cols,
                0.0f, C->data, C->cols);
}

void matrix_scale(Matrix* A, float scalar) {
    for (int i = 0; i < A->rows * A->cols; i++) {
        A->data[i] *= scalar;
    }
}

void softmax(Matrix* A) {
    for (int i = 0; i < A->rows; i++) {
        float row_max = -INFINITY;
        float sum = 0.0f;

        // Find the maximum for numerical stability
        for (int j = 0; j < A->cols; j++) {
            if (A->data[i * A->cols + j] > row_max) {
                row_max = A->data[i * A->cols + j];
            }
        }

        // Compute exponential and sum
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] = expf(A->data[i * A->cols + j] - row_max);
            sum += A->data[i * A->cols + j];
        }

        // Normalize
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] /= sum;
        }
    }
}

void scaled_dot_product_attention(Matrix* Q, Matrix* K, Matrix* V, Matrix* output, float dk_sqrt) {
    // QK^T
    Matrix scores = {(float *)malloc(Q->rows * K->rows * sizeof(float)), Q->rows, K->rows};
    matrix_multiply(Q, K, &scores);

    // Scale by sqrt(d_k)
    matrix_scale(&scores, 1.0f / dk_sqrt);

    // Apply softmax
    softmax(&scores);

    // Multiply scores by V
    matrix_multiply(&scores, V, output);

    // Free temporary allocations
    free(scores.data);
}

void multihead_attention(Matrix* Q, Matrix* K, Matrix* V,
                         Matrix* Wq, Matrix* Wk, Matrix* Wv, Matrix* Wo,
                         Matrix* Bq, Matrix* Bk, Matrix* Bv, Matrix* Bo,
                         int num_heads, Matrix* output) {
    int d_model = Q->cols;
    int d_k = d_model / num_heads;

    Matrix Q_proj = {(float *)malloc(Q->rows * d_model * sizeof(float)), Q->rows, d_model};
    Matrix K_proj = {(float *)malloc(K->rows * d_model * sizeof(float)), K->rows, d_model};
    Matrix V_proj = {(float *)malloc(V->rows * d_model * sizeof(float)), V->rows, d_model};

    // Project Q, K, V
    matrix_multiply(Q, Wq, &Q_proj);
    matrix_multiply(K, Wk, &K_proj);
    matrix_multiply(V, Wv, &V_proj);

    // Divide into heads
    Matrix Q_heads[num_heads], K_heads[num_heads], V_heads[num_heads];
    for (int i = 0; i < num_heads; i++) {
        Q_heads[i].data = Q_proj.data + i * d_k;
        Q_heads[i].rows = Q->rows;
        Q_heads[i].cols = d_k;

        K_heads[i].data = K_proj.data + i * d_k;
        K_heads[i].rows = K->rows;
        K_heads[i].cols = d_k;

        V_heads[i].data = V_proj.data + i * d_k;
        V_heads[i].rows = V->rows;
        V_heads[i].cols = d_k;
    }

    // Compute attention for each head
    Matrix attention_outputs[num_heads];
    for (int i = 0; i < num_heads; i++) {
        attention_outputs[i].data = (float *)malloc(Q->rows * d_k * sizeof(float));
        attention_outputs[i].rows = Q->rows;
        attention_outputs[i].cols = d_k;
        scaled_dot_product_attention(&Q_heads[i], &K_heads[i], &V_heads[i], &attention_outputs[i], sqrtf(d_k));
    }

    // Concatenate heads
    for (int i = 0; i < num_heads; i++) {
        memcpy(output->data + i * d_k, attention_outputs[i].data, Q->rows * d_k * sizeof(float));
        free(attention_outputs[i].data);
    }

    // Apply final projection
    Matrix temp_output = {(float *)malloc(Q->rows * d_model * sizeof(float)), Q->rows, d_model};
    matrix_multiply(output, Wo, &temp_output);
    memcpy(output->data, temp_output.data, Q->rows * d_model * sizeof(float));

    // Free memory
    free(temp_output.data);
    free(Q_proj.data);
    free(K_proj.data);
    free(V_proj.data);
}


// Main function to test multihead attention
int main() {
    int d_model = 8; // Model dimension
    int num_heads = 2;
    int seq_len = 4; // Sequence length

    // Allocate matrices
    Matrix Q = {malloc(seq_len * d_model * sizeof(float)), seq_len, d_model};
    Matrix K = {malloc(seq_len * d_model * sizeof(float)), seq_len, d_model};
    Matrix V = {malloc(seq_len * d_model * sizeof(float)), seq_len, d_model};
    Matrix Wq = {malloc(d_model * d_model * sizeof(float)), d_model, d_model};
    Matrix Wk = {malloc(d_model * d_model * sizeof(float)), d_model, d_model};
    Matrix Wv = {malloc(d_model * d_model * sizeof(float)), d_model, d_model};
    Matrix Wo = {malloc(d_model * d_model * sizeof(float)), d_model, d_model};
    Matrix Bq = {malloc(d_model * sizeof(float)), 1, d_model};
    Matrix Bk = {malloc(d_model * sizeof(float)), 1, d_model};
    Matrix Bv = {malloc(d_model * sizeof(float)), 1, d_model};
    Matrix Bo = {malloc(d_model * sizeof(float)), 1, d_model};
    Matrix output = {malloc(seq_len * d_model * sizeof(float)), seq_len, d_model};

    // Initialize random values for testing
    for (int i = 0; i < seq_len * d_model; i++) {
        Q.data[i] = K.data[i] = V.data[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < d_model * d_model; i++) {
        Wq.data[i] = Wk.data[i] = Wv.data[i] = Wo.data[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < d_model; i++) {
        Bq.data[i] = Bk.data[i] = Bv.data[i] = Bo.data[i] = (float)(rand() % 100) / 100.0f;
    }

    // Call multihead attention
    multihead_attention(&Q, &K, &V, &Wq, &Wk, &Wv, &Wo, &Bq, &Bk, &Bv, &Bo, num_heads, &output);

    // Print output
    printf("Multihead Attention Output:\n");
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            printf("%.2f ", output.data[i * output.cols + j]);
        }
        printf("\n");
    }

    // Free memory
    free(Q.data);
    free(K.data);
    free(V.data);
    free(Wq.data);
    free(Wk.data);
    free(Wv.data);
    free(Wo.data);
    free(Bq.data);
    free(Bk.data);
    free(Bv.data);
    free(Bo.data);
    free(output.data);

    return 0;
}