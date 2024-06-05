#include <opencv2/opencv.hpp>
#include <iostream>

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void toCufftComplex(const uchar* img, cufftComplex *data, int width, int height) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < width && y < height) {
        int idx = x + y*width;
        data[idx].x = static_cast<int>(img[idx]);
        data[idx].y = 0;
    }
}

class test {
public:
    test() : hostData(nullptr), deviceData(nullptr), img_back(nullptr), X(0), Y(0) {
        
    }
    ~test() {
        cufftDestroy(plan);
        if (deviceData) cudaFree(deviceData);
        if (hostData) delete[] hostData;
        if (img_back) delete img_back;
    }

    void readimg(const std::string &image_path) {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            std::cerr << "Could not read the image: " << image_path << std::endl;
            return;
        }

        X = img.cols;
        Y = img.rows;

        cudaMalloc((void**)&deviceData, sizeof(cufftComplex) * X * Y);

        dim3 block(16, 16);
        dim3 grid((X + block.x - 1) / block.x, (Y + block.y - 1) / block.y);
        toCufftComplex<<<grid, block>>>(img.ptr<uchar>(), deviceData, X, Y);
        cudaDeviceSynchronize();
    }

    void run() {
        cufftPlan2d(&plan, Y, X, CUFFT_C2C);
        cufftExecC2C(plan, deviceData, deviceData, CUFFT_FORWARD);

        hostData = new cufftComplex[X * Y];
        cudaMemcpy(hostData, deviceData, sizeof(cufftComplex) * X * Y, cudaMemcpyDeviceToHost);

        img_back = new cv::Mat(Y, X, CV_8UC1);
        for (int i = 0; i < Y; ++i) {
            for (int j = 0; j < X; ++j) {
                auto mag = sqrt(pow(hostData[i * X + j].x, 2) + pow(hostData[i * X + j].y, 2));
                img_back->at<uchar>(i, j) = static_cast<uchar>(mag*255);
            }
        }
    }

    void display() {
        cv::normalize(*img_back, *img_back, 0, 255, cv::NORM_MINMAX);

        cv::imshow("Spectrum", *img_back);
        cv::waitKey(0);
    }

private:
    
    cufftComplex *hostData;
    cufftComplex *deviceData;
    cufftHandle plan;
    int X, Y;
    cv::Mat *img_back;
};

void shiftDFT(cv::Mat &magI) {
    // Rearrange the quadrants of Fourier image so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void computeFFT(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }

    // Expand input image to optimal size
    cv::Mat padded;                            
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols); 
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Make place for both the complex and the real values
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         

    // Perform the Discrete Fourier Transform
    cv::dft(complexI, complexI);            

    // Compute the magnitude
    cv::split(complexI, planes);            
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    // Switch to logarithmic scale
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // Rearrange the quadrants of Fourier image so that the origin is at the image center
    shiftDFT(magI);

    // Normalize the magnitude image for display
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Input Image", img);    
    cv::imshow("Spectrum Magnitude", magI);
    cv::waitKey();
}

int main() {
    std::string image_path = "/home/weimin.chen/Downloads/van.jpg";
    test t;
    t.readimg(image_path);
    t.run();
    t.display();
    //computeFFT(image_path);

    return 0;
}