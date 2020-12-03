#include "RMFD.hpp"


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
RMFD::RMFD(const std::string& rawFaceFolder, const std::string& maskedFaceFolder)
{
    std::cout << "Loading Data..." << std::endl;

    load_data(rawFaceFolder, 0);
    load_data(maskedFaceFolder, 1);
    ds_size = labels.size();

    std::cout << "Data Loaded Successfully" << std::endl;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void RMFD::load_data(const std::string& folderName, const int label)
{
    const std::vector<std::string> persons = listFolders(folderName);

    const int numPersons = persons.size();

    for (const auto& person : persons)
    {
        std::vector<std::string> imageNames = listContents(person, {".jpg", ".png", ".jpeg"});

        if (!imageNames.size()==0)
        {
            images.push_back(read_image(imageNames[0]));
            labels.push_back(read_label(label));
        }
    }
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::data::Example<> RMFD::get(size_t index)
{
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::optional<size_t> RMFD::size() const
{
    return ds_size;
}
