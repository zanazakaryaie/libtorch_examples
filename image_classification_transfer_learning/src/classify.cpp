#include "utils.hpp"
#include <torch/script.h>
#include <chrono>


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
int classify_image(const std::string& imageName, torch::jit::script::Module& model, torch::nn::Linear& linear_layer)
{
    torch::Tensor img_tensor = read_image(imageName);
    img_tensor.unsqueeze_(0);

    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor temp = model.forward(input).toTensor();

    temp = temp.view({temp.size(0), -1});
    temp = linear_layer(temp);

    temp = temp.argmax(1);

    return *temp.data_ptr<long>();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc!=4)
        throw std::runtime_error("Usage: ./exe imageName modelWithoutLastLayer trainedLinearLayer");


    torch::jit::script::Module model;
    model = torch::jit::load(argv[2]);
    model.eval();

    torch::nn::Linear linear_layer(512, 2);
    torch::load(linear_layer, argv[3]);

    std::cout << "Model loaded" << std::endl;


    auto t1 = std::chrono::high_resolution_clock::now();

    int result = classify_image(argv[1], model, linear_layer);

    auto t2 = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();


    if (result==0)
        std::cout << "No Mask!!" << std::endl;
    else
        std::cout << "Has Mask :)" << std::endl;

    std::cout << "Inference time: " << duration << " ms" << std::endl;

    cv::Mat img = cv::imread(argv[1]);
    cv::imshow("input",img);
    cv::waitKey(0);


    return 0;
}
