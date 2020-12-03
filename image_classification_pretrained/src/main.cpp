#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>


//Global variables for normalization
std::vector<double> norm_mean = {0.485, 0.456, 0.406};
std::vector<double> norm_std = {0.229, 0.224, 0.225};

//Functions
std::vector<std::string> load_labels(const std::string& fileName);
torch::Tensor read_image(const std::string& imageName);
cv::Mat crop_center(const cv::Mat &img);



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc!=4)
    {
        std::cerr << "ERROR: Few input arguments!" << std::endl;
        std::cout << "Usage: ./exe modelName modelLabels imageName" << std::endl;
        return -1;
    }

    auto model = torch::jit::load(argv[1]);
    model.eval();

    std::vector<std::string> labels = load_labels(argv[2]);

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor in = read_image(argv[3]);
    inputs.push_back(in);


    auto t1 = std::chrono::high_resolution_clock::now();

    torch::Tensor output = torch::softmax(model.forward(inputs).toTensor(), 1);

    auto t2 = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << "Inference time: " << duration << " ms" << std::endl;


    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);

    torch::Tensor prob = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);

    auto probability = prob.accessor<float,1>();
    auto idx = index.accessor<long,1>();

    std::cout << "Class: " << labels[idx[0]] << std::endl;
    std::cout << "Probability: " << probability[0] << std::endl;

    cv::waitKey(0);

    return 0;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<std::string> load_labels(const std::string& fileName)
{
    std::ifstream ins(fileName);
    if (!ins.is_open())
    {
        std::cerr << "Couldn't open " << fileName << std::endl;
        abort();
    }

    std::vector<std::string> labels;
    std::string line;

    while (getline(ins, line))
        labels.push_back(line);

    ins.close();

    return labels;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::Tensor read_image(const std::string& imageName)
{
    cv::Mat img = cv::imread(imageName);
    img = crop_center(img);
    cv::resize(img, img, cv::Size(224,224));

    cv::imshow("image", img);

    if (img.channels()==1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}
