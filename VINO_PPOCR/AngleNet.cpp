#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

AngleNet::AngleNet() {}

AngleNet::~AngleNet() {
    //delete session;
    //free(inputName);
    //free(outputName);
}

void AngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //===session options===
    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    //sessionOptions.SetIntraOpNumThreads(numThread);
    //set OMP_NUM_THREADS=16

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    //sessionOptions.SetInterOpNumThreads(numThread);

    //// Sets graph optimization level
    //// ORT_DISABLE_ALL -> To disable all optimizations
    //// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    //// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    //// ORT_ENABLE_ALL -> To Enable All possible opitmizations
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

//void printInputAndOutputsInfoShort(const ov::Model& network) {
//    std::cout << "Network inputs:" << std::endl;
//    for (auto&& input : network.inputs()) {
//        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
//            << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
//            << std::endl;
//    }
//
//    std::cout << "Network outputs:" << std::endl;
//    for (auto&& output : network.outputs()) {
//        std::string out_name = "***NO_NAME***";
//        std::string node_name = "***NO_NAME***";
//
//         Workaround for "tensor has no name" issue
//        try {
//            out_name = output.get_any_name();
//        }
//        catch (const ov::Exception&) {
//        }
//        try {
//            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
//        }
//        catch (const ov::Exception&) {
//        }
//
//        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
//            << ov::layout::get_layout(output).to_string() << std::endl;
//    }
//}

void AngleNet::initModel(const std::string &pathStr) {
#ifdef _WIN32
    std::wstring clsPath = strToWstr(pathStr);
    //session = new Ort::Session(env, clsPath.c_str(), sessionOptions);
    network = ie.read_model(clsPath);

    auto device_name = "CPU";
    // -------- Step 5. Loading a model to the device --------
    executable_network = ie.compile_model(network, device_name);
#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    //getInputName(session,inputName);
    //getOutputName(session,outputName);

    //getInputName(*network, inputName);
    //getOutputName(*network, outputName);
    //printInputAndOutputsInfoShort(*network);


}

Angle scoreToAngle(const std::vector<float> &outputData) {
    int maxIndex = 0;
    float maxScore = 0;
    for (int i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return {maxIndex, maxScore};
}

Angle AngleNet::getAngle(cv::Mat &src) {

    std::vector<float> inputTensorValues = substractMeanNormalize(src, meanValues, normValues);

    std::array<int64_t, 4> inputShape{1, src.channels(), src.rows, src.cols};

    ov::element::Type input_type = ov::element::f32;
    int chanel = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    ov::Shape input_shape = { 1,3, 736, 736 };
    input_shape[0] = 1;
    input_shape[1] = chanel;
    input_shape[2] = rows;
    input_shape[3] = cols;



    float* blob = new float[inputTensorValues.size()];
    if (!inputTensorValues.empty())
    {

        memcpy(blob, &inputTensorValues[0], inputTensorValues.size() * sizeof(float));

    }
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, blob);


    const ov::Shape tensor_shape = input_tensor.get_shape();

    ov::InferRequest infer_request = executable_network.create_infer_request();

    // -------- Step 7. Prepare input --------
    infer_request.set_input_tensor(input_tensor);

    const ov::Shape tensor_shape_1 = input_tensor.get_shape();
    infer_request.infer();



    //auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    //Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
    //                                                         inputTensorValues.size(), inputShape.data(),
    //                                                         inputShape.size());
    //assert(inputTensor.IsTensor());

    //auto outputTensor = session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

    //assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    //std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    //int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
    //                                      std::multiplies<int64_t>());

    //float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    ov::Tensor outputTensors = infer_request.get_output_tensor(0);

    const ov::Shape outputShape = outputTensors.get_shape();
    auto* batchData = outputTensors.data<const float>();
    size_t count = outputTensors.get_size();
    std::vector<float> outputData(batchData, batchData + count);


    //std::vector<float> outputData(floatArray, floatArray + outputCount);
    return scoreToAngle(outputData);
}

std::vector<Angle> AngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                       const char *imgName, bool doAngle, bool mostAngle) {
    int size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
#ifdef __OPENMP__
#pragma omp parallel for num_threads(numThread)
#endif

#pragma omp parallel for num_threads(numThread)
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            cv::Mat angleImg;
            cv::resize(partImgs[i], angleImg, cv::Size(dstWidth, dstHeight));
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;

            angles[i] = angle;

            //OutPut AngleImg
            if (isOutputAngleImg) {
                std::string angleImgFile = getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}