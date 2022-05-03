#include "DbNet.h"
#include "OcrUtils.h"

DbNet::DbNet() {}

DbNet::~DbNet() {
    //free(inputName);
    //free(outputName);
}

void DbNet::setNumThread(int numOfThread) {
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

    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    bool isGPU = true;

    //std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    //auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    //OrtCUDAProviderOptions cudaOption;
    //cudaOption.device_id = 0;
    //cudaOption.arena_extend_strategy = 0;
    //cudaOption.cuda_mem_limit = 2 * 1024 * 1024 * 1024;
    //cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
    //cudaOption.do_copy_in_default_stream = 1;
    //if (isGPU && (cudaAvailable == availableProviders.end()))
    //{
    //    std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
    //    std::cout << "Inference device: CPU" << std::endl;
    //}
    //else if (isGPU && (cudaAvailable != availableProviders.end()))
    //{
    //    std::cout << "Inference device: GPU" << std::endl;
    //    sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    //}
    //else
    //{
    //    std::cout << "Inference device: CPU" << std::endl;
    //}
}

void printInputAndOutputsInfoShort(const ov::Model& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& input : network.inputs()) {
        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
            << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
            << std::endl;
    }

    std::cout << "Network outputs:" << std::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name = "***NO_NAME***";
        std::string node_name = "***NO_NAME***";

        // Workaround for "tensor has no name" issue
        try {
            out_name = output.get_any_name();
        }
        catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        }
        catch (const ov::Exception&) {
        }

        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
            << ov::layout::get_layout(output).to_string() << std::endl;
    }
}

void DbNet::initModel(const std::string &pathStr) {
#ifdef _WIN32
    std::wstring dbPath = strToWstr(pathStr);
    network = ie.read_model(dbPath);


   std:: vector<std::string> availableDevices = ie.get_available_devices();
    for (int i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
    }
    auto device_name = "CPU";
    // -------- Step 5. Loading a model to the device --------
     executable_network = ie.compile_model(network, device_name);


#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    //getInputName(*network, inputName);
    //getOutputName(*network, outputName);
    //printInputAndOutputsInfoShort(*network);
}

std::vector<TextBox> findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, ScaleParam &s,
                                 const float boxScoreThresh, const float unClipRatio) {
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / s.ratioWidth);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), s.srcWidth);

            clipMinBox[j].y = (clipMinBox[j].y / s.ratioHeight);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), s.srcHeight);
        }

        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    ov::element::Type input_type = ov::element::f32;
   int chanel = srcResize.channels();
   int rows = srcResize.rows;
   int cols = srcResize.cols;
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
    std::vector<float> output(batchData, batchData + count);
    float* floatArray = nullptr;//batchData;
    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, batchData, count * sizeof(float));

    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}
