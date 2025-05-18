#include "KeyGen.h"
#include "BlindEncrypt.h"
#include "BlindDecrypt.h"
#include "BlindEval.h"
#include "common.h"
// #define times 10000
using namespace std;
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

// 定义 CHECK 宏来检查 CUDA 错误
#define CHECK(call)                                              \
    {                                                            \
        const cudaError_t error = call;                          \
        if (error != cudaSuccess) {                              \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ \
                      << ", code: " << error                     \
                      << ", reason: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                             \
        }                                                        \
    }
#define CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while(0)

using namespace std;
using namespace cv;

struct pBox
{
	CipherText *pdata;
    //mydataFmt *pdata;
	int width;
	int height;
	// int channel;
    // int MapSize;
};

struct Weight
{
	int *pdata;
    int height;     //矩阵的高
    int width;      //矩阵的宽
};

struct Box
{
	double *pdata;
    int height;     //矩阵的高
    int width;      //矩阵的宽
};

bool readWeightData(const std::string& filename, Box& weight) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::vector<double>> matrix;  // 用于暂时存储矩阵数据

    // 读取整个文件内容
    std::string fileContent((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();

    // 处理整个文件内容，查找每一行的数字
    size_t start = 0;
    while ((start = fileContent.find('[', start)) != std::string::npos) {
        size_t end = fileContent.find(']', start);
        if (end == std::string::npos) break; // 如果没有找到结束括号，退出循环

        std::string rowStr = fileContent.substr(start + 1, end - start - 1); // 提取行数据
        std::stringstream ss(rowStr);
        std::string number;
        std::vector<double> row;

        // 分割该行的数字
        while (std::getline(ss, number, ',')) {
            // 去除字符串的空格
            number.erase(std::remove(number.begin(), number.end(), ' '), number.end());

            try {
                // 转换为 double 类型并加入行向量
                row.push_back(std::stod(number));  
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Invalid number encountered: '" << number << "' Skipping." << std::endl;
                continue; // 跳过无效数字
            } catch (const std::out_of_range&) {
                std::cerr << "Warning: Number out of range: '" << number << "' Skipping." << std::endl;
                continue; // 跳过无效数字
            }
        }

        if (!row.empty()) {
            matrix.push_back(row);  // 将该行加入矩阵
        }

        // 更新下一个行的起始位置
        start = end + 1; // 更新为下一个字符，避免死循环
    }

    // 获取矩阵的高度和宽度
    weight.height = matrix.size(); // 每行的特征数量
    if (!matrix.empty()) {
        weight.width = matrix[0].size();  // 每列的样本数量
    } else {
        std::cerr << "Error: Matrix data is empty!" << std::endl;
        return false;
    }

    // 动态分配内存来存储矩阵数据
    weight.pdata = new double[weight.height * weight.width];

    // 将数据从 vector 复制到 pdata 中
    for (int i = 0; i < weight.height; ++i) {
        for (int j = 0; j < weight.width; ++j) {
            weight.pdata[i * weight.width + j] = matrix[i][j];
        }
    }

    return true;
}

// CUDA内核函数进行矩阵转置
__global__ void transposeKernel(const int *input, int *output, int width, int height) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < width && yIndex < height) {
        output[xIndex * height + yIndex] = input[yIndex * width + xIndex];
    }
}

// CUDA转置矩阵函数
void transposeMatrix(Weight& weight) {
    // 创建转置后的数据结构
    Weight transposed;
    transposed.height = weight.width;
    transposed.width = weight.height;
    transposed.pdata = new int[transposed.height * transposed.width];

    // 在GPU上分配内存
    int *d_input, *d_output;
    cudaMalloc(&d_input, weight.height * weight.width * sizeof(int));
    cudaMalloc(&d_output, transposed.height * transposed.width * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_input, weight.pdata, weight.height * weight.width * sizeof(int), cudaMemcpyHostToDevice);

    // 设置CUDA内核的线程和块大小
    dim3 blockSize(32, 32);
    dim3 gridSize((weight.width + blockSize.x - 1) / blockSize.x, (weight.height + blockSize.y - 1) / blockSize.y);

    // 调用CUDA内核
    transposeKernel<<<gridSize, blockSize>>>(d_input, d_output, weight.width, weight.height);
    cudaDeviceSynchronize();

    // 从设备复制结果到主机
    cudaMemcpy(transposed.pdata, d_output, transposed.height * transposed.width * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(d_input);
    cudaFree(d_output);

    // 释放原始数据的内存
    delete[] weight.pdata;

    // 更新Weight结构体
    weight = transposed; // 更新为转置后的数据
}

void transposeMatrix1(Box& weight) {
    double* transposedData = new double[weight.height * weight.width];

    // 进行转置操作
    for (int i = 0; i < weight.height; ++i) {
        for (int j = 0; j < weight.width; ++j) {
            transposedData[j * weight.height + i] = weight.pdata[i * weight.width + j];
        }
    }

    // 更新 Weight 结构体中的数据
    delete[] weight.pdata;  // 释放原始数据的内存
    weight.pdata = transposedData; // 更新为转置后的数据
    std::swap(weight.height, weight.width); // 交换高度和宽度
}


void featureInit(BPS_PriParas *pri, const std::string& filename, pBox *box, struct Weight *weight) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    std::vector<double> data; // 用于暂时存储读取的数据
    std::string line;
    while (std::getline(infile, line)) {
        // 去除方括号并读取数值
        line.erase(std::remove(line.begin(), line.end(), '['), line.end());
        line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
        std::stringstream ss(line);
        double value;
        if (ss >> value) {
            data.push_back(value); // 存储数据
        }
    }

    infile.close();

    // 设置结构体的高度为1，宽度为数据的数量
    box->width = data.size(); // 宽度为数据的数量
    box->height = 1;  // 只有一行

    // 分配结构体中的pdata指针的内存并拷贝数据
    box->pdata = new CipherText[box->width]; // 注意这里分配内存大小为width
    //初始化一个中间变量
    int *temp = new int[box->width *rImageNum];
    weight->width = rImageNum;
    weight->height = data.size();
    cout<<"weight->width:"<<weight->width<<",weight->height:"<<weight->height<<endl;
    weight->pdata = new int[weight->width * weight->height];
    int cnt = 0;
    int count = 0;

    for (int i = 0; i < box->width; i++)
    {
        /* code */
        Pri_Encrypt(pri, data[i], &box->pdata[i]);
        
        // cout<<"finish"<<count++<<endl;
        for(int j=0;j<MOD_BASE_NUM;j++)
            {
                for(int k=0;k<MOD_CONFUSE_NUM;k++)
                {   
                    temp[cnt++]=box->pdata[i].getValuesOnModBase()->getvalue(j,k);
                }
            }

    }



    //转置
    cout<<box->width * rImageNum<<endl;
    int cnt1 = 0;  // 用于遍历 temp 数组的索引

    for(int i=0;i<box->width;i++)
    {
        for (int j = 0; j < rImageNum; j++)
        {
            // weight->pdata[i*box->width+j] = temp[i*box->width+j];
            weight->pdata[cnt1] = temp[cnt1];
            //cout<<weight->pdata[cnt]<<" ";
            cnt1++;
            // output_file7<<weight->pdata[i*box->width+j]<<" ";
        }
        // output_file7<<endl;
    }
    cout<<"finish";
    // output_file7.close();
}

void image2MatrixInit(BPS_PubParas *pub_paras, BPS_PriParas *pri, Mat &image, pBox *pbox, Weight *weight) {
    if (image.empty() || image.type() != CV_8UC1) {
        cout << "image's type is wrong!! Please set CV_8UC1 (single channel)" << endl;
        return;
    }
    // 将图像缩放至 28x28
    Mat resized_image;
    resize(image, resized_image, Size(28, 28)); // 确保是28x28

    pbox->height = 1;  // 只有一行
    pbox->width = resized_image.rows * resized_image.cols;  // 784 列
    cout << "image.rows: " << resized_image.rows << ", image.cols: " << resized_image.cols << endl;

    int nSize = pbox->height * pbox->width;
    // 为 pdata 分配内存
    pbox->pdata = new CipherText[nSize];
    if (pbox->pdata == NULL) {
        cout << "Memory allocation failed!" << endl;
        return;
    }

    // 初始化 pdata
    for (int i = 0; i < nSize; i++)
    {
        /* code */
        Pri_Encrypt(pri, 0, &pbox->pdata[i]);

    }
    weight->width = pbox->height * rImageNum;
    weight->height = pbox->width;
    weight->pdata = new int[iSize *rImageNum];
    for (int i = 0; i < iSize *rImageNum; i++)
    {
        /* code */
        weight->pdata[i] = 0;
    }
}



// 将图像转换为 pBox
void image2Matrix(BPS_PubParas *pub_paras, BPS_PriParas *pri, const Mat &image, const struct pBox *pbox, const struct Weight *weight){
    if ((image.data == NULL) || (image.type() != CV_8UC1)){
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    if (pbox->pdata == NULL){
        return;
    }
     // 确保图像已经被缩放为 28x28
    Mat resized_image;
    resize(image, resized_image, Size(28, 28)); // 确保是28x28
    int cnt = 0;
    //初始化一个中间变量
    int *temp = new int[iSize *rImageNum];
    int index = 0;  // 用于 pdata 索引

    for (int rowI = 0; rowI < resized_image.rows; rowI++) {
        for (int colK = 0; colK < resized_image.cols; colK++) {
            
            Pri_Encrypt(pri, (resized_image.at<uchar>(rowI, colK) / 255.0) * 2.0 - 1.0 , &pbox->pdata[index]);
            for(int j=0;j<MOD_BASE_NUM;j++)
            {
                for(int k=0;k<MOD_CONFUSE_NUM;k++)
                {   
                    temp[cnt++]=pbox->pdata[index].getValuesOnModBase()->getvalue(j,k);
                }
            }
            index++;

        }
    }
    for(int i=0;i<iSize;i++)
    {
        for (int j = 0; j < rImageNum; j++)
        {
            weight->pdata[i*rImageNum+j] = temp[i*rImageNum+j];
            // weight->pdata[j*iSize+i] = temp[i*rImageNum+j];
        }
    }
    
}

void matrixMultiplyCPU(BPS_PubParas *pub_paras, Weight *input, const Box *weight, Weight *output, int *mod) {
    // 检查输入有效性
    if (input->width != weight->height) {
        cout << "Error: Dimensions do not match for matrix multiplication!" << endl;
        return;
    }

    // 设置输出矩阵的维度
    output->height = input->height;  // 行数与输入相同
    output->width = weight->width;    // 列数与权重相同

    // 分配输出数据内存
    // output->pdata = new CipherText[output->height * output->width];
    output->pdata = new int[output->height * output->width];
    cout<<"error?"<<endl;

    if (output->pdata == NULL) {
        cout << "Error: Memory allocation for output data failed!" << endl;
        return;
    }

    // 矩阵乘法
    for (int i = 0; i < output->height; ++i) {
        for (int j = 0; j < output->width; ++j) {
            output->pdata[i * output->width + j] = 0; // 初始化为0
            for (int k = 0; k < input->width; ++k) {
                // size_t n = weight->pdata[k * weight->width + j] * 1000000;
                // int a = (static_cast<long long>(input->pdata[i * input->width + k]) * (long long)a) % mod[i / 4];
                //output->pdata[i * output->width + j] = output->pdata[i * output->width + j] + (input->pdata[i * input->width + k] * ((int)weight->pdata[k * weight->width + j]*1000000)%mod[i/4]+mod[i/4])%mod[i/4];
                long long temp = (static_cast<long long>(input->pdata[i * input->width + k]) * static_cast<long long>(weight->pdata[k * weight->width + j] * 1000000)) % mod[i / 4];
                output->pdata[i * output->width + j] = (output->pdata[i * output->width + j] + (temp + mod[i / 4]) % mod[i / 4])%mod[i / 4];
            }
            //cout << output->pdata[i * output->width + j] << " ";
        }
    }
 
}


__global__ void matrixMultiplyKernel(const int *input, const double *weight, int *output, 
                                     int inputHeight, int inputWidth, int weightWidth, int *mod) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检测
    // if (row >= inputHeight || col >= weightWidth) {
    //     printf("Thread out of bounds: row = %d, col = %d\n", row, col);
    //     return; // 如果越界，直接退出该线程的执行
    // }

    // 这里要确保线程索引在输出矩阵的范围内
    if (row < inputHeight && col < weightWidth) {
        long long temp = 0;
        output[row * weightWidth + col] = 0;

        // 执行矩阵乘法
        for (int k = 0; k < inputWidth; ++k) {
            long long inputVal = static_cast<long long>(input[row * inputWidth + k]);
            long long weightVal = static_cast<long long>(weight[k * weightWidth + col] * 1000000);
            // 逐步累加并模运算，避免溢出
            temp = (temp + (inputVal * weightVal) % mod[row / 4]) % mod[row / 4];
        }

        // 将最终计算结果存储到 output 数组
        output[row * weightWidth + col] = (temp + mod[row / 4]) % mod[row / 4];

        // 可选的调试信息，检查特定位置的输出
        if (row == 0 && col == 0) {
            printf("output[%d, %d]: %d\n", row, col, output[row * weightWidth + col]);
        }

    } 
}



void matrixMultiply2(BPS_PubParas *pub_paras, Weight *input, const Box *weight, Weight *output, 
                     int *d_input, double *d_weight, int *d_output, int *d_mod) {

    // 设置 CUDA 内核的线程和块大小
    dim3 blockSize(32, 32);
    dim3 gridSize((output->width + blockSize.x - 1) / blockSize.x, (output->height + blockSize.y - 1) / blockSize.y);
    
    cout<<d_mod<<" ";
    // 调用内核函数
    cout<<"input->width:"<<input->width<<endl;
    cout<<"weight->width:"<<weight->width<<endl;
    // 调用内核函数，将矩阵的正确维度传入内核
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_input, d_weight, d_output, 
                                              input->height, input->width, weight->width, d_mod);
    CHECK_ERROR(cudaDeviceSynchronize());
    // 在主机上创建一个数组来存储 d_mod 的值
    int mod_host[MOD_BASE_NUM];
    
    // 将 d_mod 从设备复制到主机
    cudaMemcpy(mod_host, d_mod, sizeof(int) * MOD_BASE_NUM, cudaMemcpyDeviceToHost);

    // 输出 d_mod 的值

    std::cout << "mod_host[" << 0 << "] = " << mod_host[0] << std::endl;
    
    
}


__global__ void computePolyBaseKernel(int *poly_out_pdata, int *output1_pdata, int *dev_mod1, int *poly_base_pdata, 
                                      int poly_out_height, int poly_out_width, int output1_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < poly_out_height && col < poly_out_width) {
        int a = 0;
        int s1 = 1, s2 = 1;
        int b = 1000000 % dev_mod1[row / 4];

        // 计算 s1 和 s2
        s1 = (long long)s1 * b % dev_mod1[row / 4];
        s2 = (long long)s2 * b % dev_mod1[row / 4];

        // 执行矩阵运算
        a = (long long)((poly_out_pdata[row * poly_out_width + col] * (long long)s2 * (long long)s1 % dev_mod1[row / 4])
                         + output1_pdata[row * output1_width + col] % dev_mod1[row / 4]) % dev_mod1[row / 4];

        // 存储结果
        poly_base_pdata[row * poly_out_width + col] = (a + dev_mod1[row / 4]) % dev_mod1[row / 4];
    }
}

void computePolyBaseCuda(int *d_poly_out_pdata, int *d_output1_pdata, int *d_dev_mod1, int *d_poly_base_pdata, 
                         int poly_out_height, int poly_out_width, int output1_width) {
    // 配置 CUDA 核函数的线程块和网格
    dim3 blockSize(32, 32);
    dim3 gridSize((poly_out_width + blockSize.x - 1) / blockSize.x, 
                  (poly_out_height + blockSize.y - 1) / blockSize.y);

    // 启动 CUDA 核函数
    computePolyBaseKernel<<<gridSize, blockSize>>>(d_poly_out_pdata, d_output1_pdata, d_dev_mod1, d_poly_base_pdata, 
                                                  poly_out_height, poly_out_width, output1_width);

    // 等待 GPU 完成
    cudaDeviceSynchronize();
}




int main() {

    miracl *mi = mirsys(26250, 10);
    UserKey *uk = new UserKey();
    string u = "233";
    uk->setkey(u);
    BPS_PubParas *pub = new BPS_PubParas();
    BPS_PriParas *pri = new BPS_PriParas();
    create_BPS_Paras(uk, pub, pri);
    cout << "complete successfully!" << endl;
    //显卡检测
    int dev = 3;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    Mat image = imread("/home/blind/extend/dym/Kan_Minist/data/mnist_train_1.png", IMREAD_GRAYSCALE);  // 替换为你的 MNIST 图片路径
    if (image.empty()) {
        cout << "Could not read the image!" << endl;
        return 1;
    }

    pBox input_box;
    Weight weight0;
    image2MatrixInit(pub, pri, image, &input_box, &weight0);
    cout<<"1"<<endl;
    // 转换图片到 pBox
    image2Matrix(pub, pri, image, &input_box, &weight0);
    transposeMatrix(weight0);
    // 输出转换后的数据
    cout << "input_box Height (Rows): " << input_box.height << endl;
    cout << "input_box Width (Columns): " << input_box.width << endl;

    int count = 0;

    //生成x^2
    for (int i = 0; i < weight0.height; ++i) {
        for (int j = 0; j < weight0.width; ++j) {
            weight0.pdata[i * weight0.width + j] = weight0.pdata[i * weight0.width + j] * weight0.pdata[i * weight0.width + j];
        }
    }

    // 定义Weight结构体实例
    Box weight;

    // 读取数据文件
    std::string filePath = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/base_weights.0.txt";  // 替换为你的文件路径
    //std::string filePath = "/home/blind/extend/dym/Kan_Minist/data/poly_weights_output.txt";  // 替换为你的文件路径
    readWeightData(filePath, weight);
    // 输出原始矩阵的维度
    std::cout << "Original Matrix Height (Rows): " << weight.height << std::endl;
    std::cout << "Original Matrix Width (Columns): " << weight.width << std::endl;

    // 转置矩阵
    transposeMatrix1(weight);

    // 输出转置后的矩阵的维度
    std::cout << "Transposed Matrix Height (Rows): " << weight.height << std::endl;
    std::cout << "Transposed Matrix Width (Columns): " << weight.width << std::endl;
    int *dev_mod, *dev_mod1;
    dev_mod1 = (int*)malloc(sizeof(int) * MOD_BASE_NUM);
    if (dev_mod1 == NULL) {
    std::cerr << "Error: Memory allocation for dev_mod failed!" << std::endl;
    return;
    }
    memcpy(dev_mod1, modall, sizeof(int) * MOD_BASE_NUM);

    Weight output1;
    // 设置输出矩阵的维度
    output1.height = weight0.height;  // 行数与输入相同
    output1.width = weight.width;    // 列数与权重相同

    // 分配输出数据内存
    // output->pdata = new CipherText[output->height * output->width];
    output1.pdata = new int[output1.height * output1.width];
    // 执行矩阵乘法
    Weight output1_CPU;
    // 分配GPU内存
    int *d_input1, *d_output1;
    double *d_weight1;
    cudaMalloc(&d_input1, weight0.height * weight0.width * sizeof(int));
    cudaMalloc(&d_weight1, weight.height * weight.width * sizeof(double));
    cudaMalloc(&d_output1, output1.height * output1.width * sizeof(int));
    cudaMalloc((void**)&dev_mod,sizeof(int)*MOD_BASE_NUM);

    // 将数据从主机复制到设备
    cudaMemcpy(d_input1, weight0.pdata, weight0.height * weight0.width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1, weight.pdata, weight.height * weight.width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mod,modall,sizeof(int)*MOD_BASE_NUM,cudaMemcpyHostToDevice);
    auto start2 = std::chrono::high_resolution_clock::now();
    matrixMultiply2(pub,  &weight0, &weight, &output1, d_input1, d_weight1, d_output1, dev_mod);
    cout<<"GPU finish"<<endl;
    //matrixMultiply2(pub, &weight0, &weight, &output1, dev_mod);
    // matrixMultiplyCPU(pub, &weight0, &weight, &output1_CPU, dev_mod1);
    auto end2 = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    cout<<"CPU finish"<<endl;
      // 从设备复制结果到主机
    cudaMemcpy(output1.pdata, d_output1, output1.height * output1.width * sizeof(int), cudaMemcpyDeviceToHost);
    // 在主程序中释放 GPU 内存
    cudaFree(d_input1);
    cudaFree(d_weight1);
    cudaFree(d_output1);
    cout<<"----------------"<<endl;

    //第一次比较
    cout<<"--------------"<<endl;
    //计算ploy_out
    std::string fileName3 = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/legendre_bias0_output.txt";  // 替换为你的文件路径
    pBox legendre_1;
    Weight weight1;
    featureInit(pri, fileName3, &legendre_1, &weight1);
    transposeMatrix(weight1);
    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "weight1 Width: " << weight1.width << ", weight1 Height: " << weight1.height << std::endl;
    Box poly_weight0;
    // 读取数据文件
    std::string fileName4 = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/poly_weights.0.txt";  // 替换为你的文件路径
    //std::string filePath = "/home/blind/extend/dym/Kan_Minist/data/poly_weights_output.txt";  // 替换为你的文件路径
    readWeightData(fileName4, poly_weight0);
    // 转置矩阵
    transposeMatrix1(poly_weight0);

    // 输出转置后的矩阵的维度
    std::cout << "Transposed poly_weight0 Height (Rows): " << poly_weight0.height << std::endl;
    std::cout << "Transposed poly_weight0 Width (Columns): " << poly_weight0.width << std::endl;
    // 初始化输出 pBox
    Weight poly_out;
    poly_out.height = weight1.height;  // 行数与输入相同
    poly_out.width = poly_weight0.width;    // 列数与权重相同

    // 分配输出数据内存
    // output->pdata = new CipherText[output->height * output->width];
    poly_out.pdata = new int[poly_out.height * poly_out.width];
    // 执行矩阵乘法
    Weight poly_out_CPU;
    // 分配GPU内存
    int *d_input2, *d_output2, *dev_mod2;
    double *d_weight2;
    cudaMalloc(&d_input2, weight1.height * weight1.width * sizeof(int));
    cudaMalloc(&d_weight2, poly_weight0.height * poly_weight0.width * sizeof(double));
    cudaMalloc(&d_output2, poly_out.height * poly_out.width * sizeof(int));
    //cudaMalloc((void**)&dev_mod2,sizeof(int)*MOD_BASE_NUM);

    // 将数据从主机复制到设备
    cudaMemcpy(d_input2, weight1.pdata, weight1.height * weight1.width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, poly_weight0.pdata, poly_weight0.height * poly_weight0.width * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_mod,modall,sizeof(int)*MOD_BASE_NUM,cudaMemcpyHostToDevice);
    auto start3 = std::chrono::high_resolution_clock::now();
    matrixMultiply2(pub,  &weight1, &poly_weight0, &poly_out, d_input2, d_weight2, d_output2, dev_mod);
    // 记录结束时间
    auto end3 = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count();
    cout<<"GPU finish"<<endl;
    // matrixMultiply2(pub, &weight1, &poly_weight0, &poly_out, dev_mod);
    // matrixMultiplyCPU(pub, &weight1, &poly_weight0, &poly_out_CPU, dev_mod1);
    cudaMemcpy(poly_out.pdata, d_output2, poly_out.height * poly_out.width * sizeof(int), cudaMemcpyDeviceToHost);
    // 在主程序中释放 GPU 内存
    cudaFree(d_input2);
    cudaFree(d_weight2);
    cudaFree(d_output2);
    
    //cudaFree(dev_mod2);
    cout<<"--------------"<<endl;
    cout<<"the seond compare:";
    int count0 = 0;

    cout<<endl;
    cout<<poly_out.pdata[0]<<endl;
    cout<<poly_out_CPU.pdata[0]<<endl;;
    cout<<"count0:"<<count0<<endl;

    cout << "Output Matrix Height: " << poly_out.height << ", Width: " << poly_out.width << endl;


    auto start4 = std::chrono::high_resolution_clock::now();
    Weight poly_base_1;
    // computePolyBase(poly_base_1, poly_out, output1, dev_mod);
    poly_base_1.height = poly_out.height;
    poly_base_1.width = poly_out.width;
    poly_base_1.pdata = new int[poly_out.height * poly_out.width];
    cout<<" poly_base_1.height:"<< poly_base_1.height <<", poly_base_1.width"<< poly_base_1.width<<endl;

    int s1, s2, b;
    for (int i = 0; i < poly_base_1.height; ++i) {
        for (int j = 0; j < poly_base_1.width; ++j) {
            int a = 0;
            s1 = s2 = 1;
            b = 1000000 % dev_mod1[i / 4];
            s1 = (long long)s1 * b % dev_mod1[i / 4];
            s2 = (long long)s2 * b % dev_mod1[i / 4];
            a = (long long)((poly_out.pdata[i * poly_out.width + j] * (long long)s2 * (long long)s1 % dev_mod1[i / 4]) + output1.pdata[i * output1.width + j]  % dev_mod1[i / 4]) % dev_mod1[i / 4];
            //poly_base_1.pdata[i * poly_base_1.width + j] = ((poly_out.pdata[i * poly_out.width + j] * 1000000 % dev_mod[i / 4])% dev_mod[i / 4] + output1.pdata[i * output1.width + j]  % dev_mod[i / 4]) % dev_mod[i / 4];
            //cout << poly_base_1.pdata[i * poly_base_1.width + j] << " ";
            poly_base_1.pdata[i * poly_base_1.width + j] = (a + dev_mod1[i / 4]) % dev_mod1[i / 4];
        }
        //cout << endl;
    }
     // 记录结束时间
    auto end4 = std::chrono::high_resolution_clock::now();

    // 计算持续时间
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4).count();



    //计算第二层
    Box base_weight1;
    // 读取数据文件
    std::string fileName5 = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/base_weights.1.txt";  // 替换为你的文件路径
    //std::string filePath = "/home/blind/extend/dym/Kan_Minist/data/poly_weights_output.txt";  // 替换为你的文件路径
    readWeightData(fileName5, base_weight1);
    // 转置矩阵
    transposeMatrix1(base_weight1);

    // 输出转置后的矩阵的维度
    std::cout << "Transposed base_weight1 Height (Rows): " << base_weight1.height << std::endl;
    std::cout << "Transposed base_weight1 Width (Columns): " << base_weight1.width << std::endl;
    // 初始化输出 pBox
    Weight base_out2;
    base_out2.height = poly_base_1.height;  // 行数与输入相同
    base_out2.width = base_weight1.width;    // 列数与权重相同

    // 分配输出数据内存
    // output->pdata = new CipherText[output->height * output->width];
    base_out2.pdata = new int[base_out2.height * base_out2.width];
    //cudaFree(dev_mod);
    Weight base_out2_CPU;
    int *d_input3, *d_output3, *dev_mod3;
    double *d_weight3;
    CHECK(cudaMalloc(&d_input3, poly_base_1.height * poly_base_1.width * sizeof(int)));
    CHECK(cudaMalloc(&d_weight3, base_weight1.height * base_weight1.width * sizeof(double)));
    CHECK(cudaMalloc(&d_output3, base_out2.height * base_out2.width * sizeof(int)));
    CHECK(cudaMemset(d_output3, 0, base_out2.height * base_out2.width * sizeof(int)));

    //CHECK(cudaMalloc((void**)&dev_mod3,sizeof(int)*MOD_BASE_NUM));

    // 将数据从主机复制到设备
    CHECK(cudaMemcpy(d_input3, poly_base_1.pdata, poly_base_1.height * poly_base_1.width * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight3, base_weight1.pdata, base_weight1.height * base_weight1.width * sizeof(double), cudaMemcpyHostToDevice));
    auto start5 = std::chrono::high_resolution_clock::now();
    matrixMultiply2(pub,  &poly_base_1, &base_weight1, &base_out2, d_input3, d_weight3, d_output3, dev_mod);
    // 记录结束时间
    auto end5 = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end5 - start5).count();
    cout<<"GPU finish"<<endl;
   
    // 执行矩阵乘法
    //matrixMultiply2(pub, &poly_base_1, &base_weight1, &base_out2, dev_mod);
    // matrixMultiplyCPU(pub, &poly_base_1, &base_weight1, &base_out2_CPU, dev_mod1);
    cout<<"base_out2_CPU width:"<<base_out2_CPU.width<<", base_out2_CPU height:"<<base_out2_CPU.height<<endl;
    cout<<"base_out2 width:"<<base_out2.width<<", base_out2 height:"<<base_out2.height<<endl;

    CHECK_ERROR(cudaMemcpy(base_out2.pdata, d_output3, base_out2.height * base_out2.width * sizeof(int), cudaMemcpyDeviceToHost));
    // 在主程序中释放 GPU 内存
    cudaFree(d_input3);
    cudaFree(d_weight3);
    cudaFree(d_output3);
    //cudaFree(dev_mod3);

    cout<<"--------------"<<endl;
    cout<<"the third compare:";
    int count1 = 0;

    cout<<"count1:"<<count1<<endl;



    cout << "base_out2 Matrix Height: " << base_out2.height << ", Width: " << base_out2.width << endl;

    
    pBox poly_base_out;
    poly_base_out.width = base_out2_CPU.width;
    poly_base_out.height = 1;
    poly_base_out.pdata = new CipherText[poly_base_out.width * poly_base_out.height];
    //std::ofstream output_file11("/home/blind/extend/dym/RT-HCNN/data1/poly_base_out.txt");  // 创建输出文件

    for (int i = 0; i < poly_base_out.width; i++)
    {
        /* code */
        Pri_Encrypt(pri, 0, &poly_base_out.pdata[i]);
    }
    

    for (int i = 0; i < base_out2_CPU.width; ++i)
    {
        /* code */
        for(int j=0;j<MOD_BASE_NUM;j++)
        {
                for(int k=0;k<MOD_CONFUSE_NUM;k++)
                {   
                    int index = (j * MOD_CONFUSE_NUM + k) * base_out2.width + i; 
                    poly_base_out.pdata[i].getValuesOnModBase()->setvalue(j, k, base_out2.pdata[index]);
                    //cout<<weight0.pdata[j * weight0.width + k]<<" ";
                }
        }
        double ans = Decrypt(pri, &poly_base_out.pdata[i]);
        cout<<ans<<" ";
    }

    
     //计算ploy_out
    std::string fileName6 = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/legendre_bias1_output.txt";  // 替换为你的文件路径
    pBox legendre_2;
    Weight weight2;
    featureInit(pri, fileName6, &legendre_2, &weight2);
    transposeMatrix(weight2);

    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "weight2 Width: " << weight2.width << ", weight2 Height: " << weight2.height << std::endl;

    Box poly_weight1;
    // 读取数据文件
    std::string fileName7 = "/home/blind/extend/dym/Pytorch-MTCNN/test/order_3/model_params/poly_weights.1.txt";  // 替换为你的文件路径
    //std::string filePath = "/home/blind/extend/dym/Kan_Minist/data/poly_weights_output.txt";  // 替换为你的文件路径
    readWeightData(fileName7, poly_weight1);
    // 转置矩阵
    transposeMatrix1(poly_weight1);

    // 输出转置后的矩阵的维度
    std::cout << "Transposed poly_weight1 Height (Rows): " << poly_weight1.height << std::endl;
    std::cout << "Transposed poly_weight1 Width (Columns): " << poly_weight1.width << std::endl;
    // 初始化输出 pBox
    Weight poly_out2;
    poly_out2.height = weight2.height;  // 行数与输入相同
    poly_out2.width = poly_weight1.width;    // 列数与权重相同

    // 分配输出数据内存
    // output->pdata = new CipherText[output->height * output->width];
    poly_out2.pdata = new int[poly_out2.height * poly_out2.width];
    Weight poly_out2_CPU;
    int *d_input4, *d_output4, *dev_mod4;
    double *d_weight4;
    cudaMalloc(&d_input4, weight2.height * weight2.width * sizeof(int));
    cudaMalloc(&d_weight4, poly_weight1.height * poly_weight1.width * sizeof(double));
    cudaMalloc(&d_output4, poly_out2.height * poly_out2.width * sizeof(int));
    //cudaMalloc((void**)&dev_mod2,sizeof(int)*MOD_BASE_NUM);

    // 将数据从主机复制到设备
    cudaMemcpy(d_input4, weight2.pdata, weight2.height * weight2.width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight4, poly_weight1.pdata, poly_weight1.height * poly_weight1.width * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_mod,modall,sizeof(int)*MOD_BASE_NUM,cudaMemcpyHostToDevice);
    auto start6 = std::chrono::high_resolution_clock::now();
    matrixMultiply2(pub,  &weight2, &poly_weight1, &poly_out2, d_input4, d_weight4, d_output4, dev_mod);
    auto end6 = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration6 = std::chrono::duration_cast<std::chrono::milliseconds>(end6 - start6).count();
    cout<<"GPU finish"<<endl;
    // 执行矩阵乘法
    // matrixMultiply2(pub, &weight2, &poly_weight1, &poly_out2, dev_mod);
    // matrixMultiplyCPU(pub, &weight2, &poly_weight1, &poly_out2_CPU, dev_mod1);
     CHECK_ERROR(cudaMemcpy(poly_out2.pdata, d_output4, poly_out2.height * poly_out2.width * sizeof(int), cudaMemcpyDeviceToHost));
    // 在主程序中释放 GPU 内存
    cudaFree(d_input4);
    cudaFree(d_weight4);
    cudaFree(d_output4);
    
    int count2 = 0;

    cout<<"count2:"<<count2<<endl;
    cout<<poly_out2.pdata[0]<<endl;


    auto start7 = std::chrono::high_resolution_clock::now();
    Weight poly_base_out2;
    poly_base_out2.height = poly_out2.height;
    poly_base_out2.width = poly_out2.width;
    poly_base_out2.pdata = new int[poly_base_out2.height * poly_base_out2.width];
    cout<<" poly_base_out2.height:"<< poly_base_out2.height <<", poly_base_out2.width"<< poly_base_out2.width<<endl;
    int s11, s22, b1;
    for (int i = 0; i < poly_base_out2.height; ++i) {
        for (int j = 0; j < poly_base_out2.width; ++j) {
            int a = 0;
            s11 = s22 = 1;
            b1 = 1000000 % dev_mod1[i / 4];
            s11 = (long long)s11 * b1 % dev_mod1[i / 4];
            s22 = (long long)s22 * b1 * b1 % dev_mod1[i / 4];
            a = (long long)((poly_out2.pdata[i * poly_out2.width + j] * (long long)s22 * (long long)s11 % dev_mod1[i / 4]) + base_out2.pdata[i * base_out2.width + j]  % dev_mod1[i / 4]) % dev_mod1[i / 4];
            //poly_base_1.pdata[i * poly_base_1.width + j] = ((poly_out.pdata[i * poly_out.width + j] * 1000000 % dev_mod[i / 4])% dev_mod[i / 4] + output1.pdata[i * output1.width + j]  % dev_mod[i / 4]) % dev_mod[i / 4];
            //cout << poly_base_1.pdata[i * poly_base_1.width + j] << " ";
            poly_base_out2.pdata[i * poly_base_out2.width + j] = (a + dev_mod1[i / 4]) % dev_mod1[i / 4];
        }
        //cout << endl;
    }
    auto end7 = std::chrono::high_resolution_clock::now();

    // 计算持续时间
    auto duration7 = std::chrono::duration_cast<std::chrono::milliseconds>(end7 - start7).count();

    pBox poly_base_out3;
    poly_base_out3.width = poly_base_out2.width;
    poly_base_out3.height = 1;
    cout<<poly_base_out3.width * poly_base_out3.height<<endl;
    poly_base_out3.pdata = new CipherText[poly_base_out3.width * poly_base_out3.height];
    if (poly_base_out3.pdata == NULL) {
    // 处理内存分配失败
        cout<<"error"<<endl;
    }
    
    for (int i = 0; i < poly_base_out3.width; i++)
    {
        /* code */
        Pri_Encrypt(pri, 0, &poly_base_out3.pdata[i]);
    }
    cout<<"1"<<endl;

    for (int i = 0; i < poly_base_out3.width; ++i)
    {
        /* code */
        for(int j=0;j<MOD_BASE_NUM;j++)
        {
                for(int k=0;k<MOD_CONFUSE_NUM;k++)
                {   
                    int index = (j * MOD_CONFUSE_NUM + k) * poly_base_out2.width + i; 
                    poly_base_out3.pdata[i].getValuesOnModBase()->setvalue(j, k, poly_base_out2.pdata[index]);
                    //cout<<weight0.pdata[j * weight0.width + k]<<" ";
                }
        }
        double ans = Decrypt(pri, &poly_base_out3.pdata[i]);
        cout<<ans/ 1e+24<<" ";
    }

    double totalTime =  duration2 + duration3 + duration4 + duration5 + duration6 + duration7;
    std::cout << "time: " << totalTime << "ms" << std::endl;
 

    return 0;
}


