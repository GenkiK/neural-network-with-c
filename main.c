#include "nn.h"

// helper
void copy_arr(int n, float const *src, float *tgt) {
    for (int i = 0; i < n; i++) {
        tgt[i] = src[i];
    }
}

void print_sum(int n, float *x) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    printf("%f\n", sum);
}

// 1
void print(int m, int n, const float *x) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", x[i * n + j]);
        }
        printf("\n");
    }
}

// 2
void fc(int m, int n, const float *x, const float *A, const float *b,
        float *y) {
    // A: [m, n]
    // b: [m,]
    // x: [n,]
    // y: [m,]
    float sum;
    for (int i = 0; i < m; i++) {
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        sum += b[i];
        y[i] = sum;
    }
}

// 3
void relu(int n, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] > 0.0 ? x[i] : 0.0;
    }
}

// 4
void softmax(int n, const float *x, float *y) {
    float denom = 0.0;
    float maxv = FLT_MIN;

    for (int i = 0; i < n; i++) {
        maxv = x[i] > maxv ? x[i] : maxv;
    }
    for (int i = 0; i < n; i++) {
        denom += exp(x[i] - maxv);
    }
    for (int i = 0; i < n; i++) {
        y[i] = exp(x[i] - maxv) / denom;
    }
}

// 5
int inference3(int m, int n, const float *A, const float *b, const float *x,
               float *y) {
    fc(m, n, x, A, b, y);
    relu(m, y, y);
    softmax(m, y, y);
    int max_idx = 0;
    for (int i = 1; i < m; i++) {
        max_idx = y[i] > y[max_idx] ? i : max_idx;
    }
    return max_idx;
}
int inference6(int m1, int n1, int m2, int n2, int m3, int n3, const float *A1,
               const float *A2, const float *A3, const float *b1,
               const float *b2, const float *b3, const float *x, float *y1,
               float *y2, float *y3) {
    fc(m1, n1, x, A1, b1, y1);
    relu(m1, y1, y1);
    fc(m2, n2, y1, A2, b2, y2);
    relu(m2, y2, y2);
    fc(m3, n3, y2, A3, b3, y3);
    softmax(m3, y3, y3);
    int max_idx = 0;
    for (int i = 1; i < m3; i++) {
        max_idx = y3[i] > y3[max_idx] ? i : max_idx;
    }
    return max_idx;
}

// 8
void softmaxwithloss_bwd(int m, const float *y, unsigned char t, float *dEdx) {
    for (int j = 0; j < m; j++) {
        if (t == j) {
            dEdx[j] = y[j] - 1;
        } else {
            dEdx[j] = y[j];
        }
    }
}

// 9
void relu_bwd(int m, const float *x, const float *dEdy, float *dEdx) {
    for (int j = 0; j < m; j++) {
        dEdx[j] = x[j] > 0 ? dEdy[j] : 0.0;
    }
}

// 10
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
        dEdb[i] = dEdy[i];
    }

    float dEdx_tmp;
    for (j = 0; j < n; j++) {
        dEdx_tmp = 0.0;
        for (i = 0; i < m; i++) {
            dEdx_tmp += A[i * n + j] * dEdy[i];
        }
        dEdx[j] = dEdx_tmp;
    }
}

// 11
void backward3(int m, int n, const float *A, const float *b, const float *x,
               unsigned char t, float *y, float *dEdA, float *dEdb) {
    float *dEdx = malloc(sizeof(float) * n);
    float *relu_input = malloc(sizeof(float) * m);

    // forward
    fc(m, n, x, A, b, y);
    copy_arr(m, y, relu_input);
    relu(m, y, y);
    softmax(m, y, y);

    // backward
    softmaxwithloss_bwd(m, y, t, dEdx);
    relu_bwd(m, relu_input, dEdx, dEdx);
    fc_bwd(m, n, x, dEdx, A, dEdA, dEdb, dEdx);

    free(dEdx);
    free(relu_input);
}

// 16
void backward6(int m1, int n1, int m2, int n2,  // m1 == n2
               int m3, int n3,                  // m2 == n3
               const float *A1, const float *A2, const float *A3,
               const float *b1, const float *b2, const float *b3,
               const float *x, unsigned char t, float *y1, float *y2, float *y3,
               float *dEdA1, float *dEdA2, float *dEdA3, float *dEdb1,
               float *dEdb2, float *dEdb3) {
    // W: [50, 784] -> [100, 50] -> [10, 100]
    float *relu1_input = malloc(sizeof(float) * m1);
    float *relu2_input = malloc(sizeof(float) * m2);

    float *dEdx3 = malloc(sizeof(float) * m3);
    float *dEdx2 = malloc(sizeof(float) * m2);
    float *dEdx1 = malloc(sizeof(float) * m1);
    float *dEdx0 = malloc(sizeof(float) * n1);

    // forward
    fc(m1, n1, x, A1, b1, y1);
    copy_arr(m1, y1, relu1_input);
    relu(m1, y1, y1);
    fc(m2, n2, y1, A2, b2, y2);
    copy_arr(m2, y2, relu2_input);
    relu(m2, y2, y2);
    fc(m3, n3, y2, A3, b3, y3);
    softmax(m3, y3, y3);

    // backward
    softmaxwithloss_bwd(m3, y3, t, dEdx3);
    fc_bwd(m3, n3, y2, dEdx3, A3, dEdA3, dEdb3, dEdx2);
    relu_bwd(m2, relu2_input, dEdx2, dEdx2);
    fc_bwd(m2, n2, y1, dEdx2, A2, dEdA2, dEdb2, dEdx1);
    relu_bwd(n1, relu1_input, dEdx1, dEdx1);
    fc_bwd(m1, n1, x, dEdx1, A1, dEdA1, dEdb1, dEdx0);

    free(dEdx0);
    free(dEdx1);
    free(dEdx2);
    free(dEdx3);
    free(relu1_input);
    free(relu2_input);
}

// 12
void swap(int i, int j, int *arr) {
    int tmp = arr[j];
    arr[j] = arr[i];
    arr[i] = tmp;
}
void shuffle(int n, int *x) {
    for (int i = 0; i < n; i++) {
        int j = (int)((float)(rand()) / RAND_MAX * (n - 1));
        swap(i, j, x);
    }
}

// 13
float cross_entropy_error(const float *y, int t) { return -log(y[t] + 1e-7); }

// 14
void add(int n, const float *x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] += x[i];
    }
}
void scale(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }
}
void init(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = x;
    }
}
void rand_init(int n, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = (float)(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

// 15
void train3(int m, int n, float *A, float *b, float *train_x,
            unsigned char *train_y, float *test_x, unsigned char *test_y,
            int train_count, int test_count, int n_epoch, int bs, float lr) {
    int mn = m * n;
    rand_init(mn, A);
    rand_init(m, b);

    int *index = malloc(sizeof(int) * train_count);

    float *m_dEdA = malloc(sizeof(float) * mn);
    float *m_dEdb = malloc(sizeof(float) * m);

    float *dEdA = malloc(sizeof(float) * mn);  // no need to init
    float *dEdb = malloc(sizeof(float) * m);   // no need to init
    float *y = malloc(sizeof(float) * m);

    float sum_loss;
    // float sum_test_loss;

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        for (int i = 0; i < train_count; i++) {
            index[i] = i;
        }
        shuffle(train_count, index);
        sum_loss = 0.0;
        for (int iter = 0; iter < train_count / bs; iter++) {
            init(mn, 0.0, m_dEdA);
            init(m, 0.0, m_dEdb);
            for (int batch_idx = 0; batch_idx < bs; batch_idx++) {
                int idx = index[bs * iter + batch_idx];
                backward3(m, n, A, b, train_x + n * idx, train_y[idx], y, dEdA,
                          dEdb);
                sum_loss += cross_entropy_error(y, train_y[idx]);
                add(mn, dEdA, m_dEdA);
                add(m, dEdb, m_dEdb);
            }  // end of 1iter
            // mean and update params
            scale(mn, -lr / bs, m_dEdA);
            scale(m, -lr / bs, m_dEdb);
            add(mn, m_dEdA, A);
            add(m, m_dEdb, b);
        }  // end of 1epoch
        printf("mean train loss:\t%.4f\n", sum_loss / train_count);

        // inference for test data
        float sum_test_loss = 0.0;
        int acc_sum = 0;
        for (int i = 0; i < test_count; i++) {
            if (inference3(m, n, A, b, test_x + i * n, y) == test_y[i]) {
                acc_sum++;
            }
            sum_test_loss += cross_entropy_error(y, test_y[i]);
        }
        printf("mean test loss:\t\t%.4f\n", sum_test_loss / test_count);
        printf("test acc:\t\t%f%%\n", acc_sum * 100.0 / test_count);
    }
    free(index);
    free(m_dEdA);
    free(m_dEdb);
    free(dEdA);
    free(dEdb);
    free(y);
}

// 15
void train6(int m1, int n1, int m2, int n2, int m3, int n3, float *A1,
            float *A2, float *A3, float *b1, float *b2, float *b3,
            float *train_x, unsigned char *train_y, float *test_x,
            unsigned char *test_y, int train_count, int test_count, int n_epoch,
            int bs, float lr) {
    int mn1 = m1 * n1;
    int mn2 = m2 * n2;
    int mn3 = m3 * n3;
    rand_init(mn1, A1);
    rand_init(m1, b1);
    rand_init(mn2, A2);
    rand_init(m2, b2);
    rand_init(mn3, A3);
    rand_init(m3, b3);

    int *index = malloc(sizeof(int) * train_count);

    float *m_dEdA1 = malloc(sizeof(float) * mn1);
    float *m_dEdb1 = malloc(sizeof(float) * m1);
    float *m_dEdA2 = malloc(sizeof(float) * mn2);
    float *m_dEdb2 = malloc(sizeof(float) * m2);
    float *m_dEdA3 = malloc(sizeof(float) * mn3);
    float *m_dEdb3 = malloc(sizeof(float) * m3);

    float *dEdA1 = malloc(sizeof(float) * mn1);
    float *dEdb1 = malloc(sizeof(float) * m1);
    float *y1 = malloc(sizeof(float) * m1);
    float *dEdA2 = malloc(sizeof(float) * mn2);
    float *dEdb2 = malloc(sizeof(float) * m2);
    float *y2 = malloc(sizeof(float) * m2);
    float *dEdA3 = malloc(sizeof(float) * mn3);
    float *dEdb3 = malloc(sizeof(float) * m3);
    float *y3 = malloc(sizeof(float) * m3);

    float sum_loss;

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        for (int i = 0; i < train_count; i++) {
            index[i] = i;
        }
        shuffle(train_count, index);
        sum_loss = 0.0;
        for (int iter = 0; iter < train_count / bs; iter++) {
            init(mn3, 0.0, m_dEdA3);
            init(m3, 0.0, m_dEdb3);
            init(mn2, 0.0, m_dEdA2);
            init(m2, 0.0, m_dEdb2);
            init(mn1, 0.0, m_dEdA1);
            init(m1, 0.0, m_dEdb1);
            for (int batch_idx = 0; batch_idx < bs; batch_idx++) {
                int idx = index[bs * iter + batch_idx];
                // backward3(m, n, A, b, train_x + n * idx, train_y[idx], y,
                // dEdA, dEdb);
                backward6(m1, n1, m2, n2, m3, n3, A1, A2, A3, b1, b2, b3,
                          train_x + n1 * idx, train_y[idx], y1, y2, y3, dEdA1,
                          dEdA2, dEdA3, dEdb1, dEdb2, dEdb3);
                sum_loss += cross_entropy_error(y3, train_y[idx]);
                add(mn1, dEdA1, m_dEdA1);
                add(m1, dEdb1, m_dEdb1);
                add(mn2, dEdA2, m_dEdA2);
                add(m2, dEdb2, m_dEdb2);
                add(mn3, dEdA3, m_dEdA3);
                add(m3, dEdb3, m_dEdb3);
            }  // end of 1iter
            // mean and update params
            scale(mn1, -lr / bs, m_dEdA1);
            scale(m1, -lr / bs, m_dEdb1);
            add(mn1, m_dEdA1, A1);
            add(m1, m_dEdb1, b1);

            scale(mn2, -lr / bs, m_dEdA2);
            scale(m2, -lr / bs, m_dEdb2);
            add(mn2, m_dEdA2, A2);
            add(m2, m_dEdb2, b2);

            scale(mn3, -lr / bs, m_dEdA3);
            scale(m3, -lr / bs, m_dEdb3);
            add(mn3, m_dEdA3, A3);
            add(m3, m_dEdb3, b3);
        }  // end of 1epoch
        printf("mean train loss:\t%.4f\n", sum_loss / train_count);

        // inference for test data
        float sum_test_loss = 0.0;
        int acc_sum = 0;
        for (int i = 0; i < test_count; i++) {
            if (inference6(m1, n1, m2, n2, m3, n3, A1, A2, A3, b1, b2, b3,
                           test_x + n1 * i, y1, y2, y3) == test_y[i]) {
                acc_sum++;
            }
            sum_test_loss += cross_entropy_error(y3, test_y[i]);
        }
        printf("mean test loss:\t\t%.4f\n", sum_test_loss / test_count);
        printf("test acc:\t\t%f%%\n", acc_sum * 100.0 / test_count);
    }
    free(index);
    free(m_dEdA1);
    free(m_dEdb1);
    free(dEdA1);
    free(dEdb1);
    free(y1);
    free(m_dEdA2);
    free(m_dEdb2);
    free(dEdA2);
    free(dEdb2);
    free(y2);
    free(m_dEdA3);
    free(m_dEdb3);
    free(dEdA3);
    free(dEdb3);
    free(y3);
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

void load(const char *filename, int m, int n, float *A, float *b) {
    FILE *fp;
    fp = fopen(filename, "r");
    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
}

int main() {
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count,
               &width, &height);

    // W: [50, 784] -> [100, 50] -> [10, 100]
    int m1 = 50;
    int n1 = width * height;
    int m2 = 100;
    int n2 = m1;
    int m3 = 10;
    int n3 = m2;

    int n_epoch = 10;
    int bs = 100;
    float lr = 0.1;
    float *A1 = malloc(sizeof(float) * m1 * n1);
    float *b1 = malloc(sizeof(float) * m1);
    float *A2 = malloc(sizeof(float) * m2 * n2);
    float *b2 = malloc(sizeof(float) * m2);
    float *A3 = malloc(sizeof(float) * m3 * n3);
    float *b3 = malloc(sizeof(float) * m3);
    train6(m1, n1, m2, n2, m3, n3, A1, A2, A3, b1, b2, b3, train_x, train_y,
           test_x, test_y, train_count, test_count, n_epoch, bs, lr);
    return 0;
}