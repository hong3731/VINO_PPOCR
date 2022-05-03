#include "CrnnNet.h"
#include "OcrUtils.h"
#include <fstream>
#include <numeric>

CrnnNet::CrnnNet() {}

CrnnNet::~CrnnNet() {
    //delete session;
    //free(inputName);
    //free(outputName);
}

void CrnnNet::setNumThread(int numOfThread) {
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

    //bool isGPU = true;

    //std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    //auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    //OrtCUDAProviderOptions cudaOption;
    //cudaOption.device_id = 0;
    //cudaOption.arena_extend_strategy = 0;
    //cudaOption.cuda_mem_limit = 1 * 1024 * 1024 * 1024;
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
//        // Workaround for "tensor has no name" issue
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
void CrnnNet::initModel(const std::string &pathStr, const std::string &keysPath) {
#ifdef _WIN32
    std::wstring dbPath = strToWstr(pathStr);
    network = ie.read_model(dbPath);

    auto device_name = "CPU";
    // -------- Step 5. Loading a model to the device --------
    executable_network = ie.compile_model(network, device_name);


#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    //getInputName(*network, inputName);
    //getOutputName(*network, outputName);
    //printInputAndOutputsInfoShort(*network);

    //load keys
    std::ifstream in(keysPath.c_str());
    std::string line;
    if (in) {
        while (getline(in, line)) {// line中不包括每行的换行符
            keys.push_back(line);
        }
    } else {
        printf("The keys.txt file was not found\n");
        return;
    }
    /*if (keys.size() != 6623) {
        fprintf(stderr, "missing keys\n");
    }*/
    keys.insert(keys.begin(), "#");
    keys.emplace_back(" ");
    printf("total keys size(%lu)\n", keys.size());
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine CrnnNet::scoreToTextLine(const std::vector<float> &outputData, int h, int w) {
    int keySize = keys.size();
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;

    for (int i = 0; i < h; i++) {
        maxIndex = int(argmax(&outputData[i * w], &outputData[(i + 1) * w - 1]));
        maxValue = float(*std::max_element(&outputData[i * w], &outputData[(i + 1) * w - 1]));

        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores};
}

TextLine CrnnNet::getTextLine(const cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);

    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));

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
    std::vector<float> outputData(batchData, batchData + count);


    //std::vector<float> outputData(floatArray, floatArray + outputCount);
    return scoreToTextLine(outputData, outputShape[1], outputShape[2]);
}

std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
#ifdef __OPENMP__
#pragma omp parallel for num_threads(numThread)
#endif
#pragma omp parallel for num_threads(numThread)
    for (int i = 0; i < size; ++i) {
        //OutPut DebugImg
        if (isOutputDebugImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
            saveImg(partImg[i], debugImgFile.c_str());
        }

        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}