// VINO_PPOCR.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "OcrLite.h"
#include "OcrUtils.h"





int main()
{
    std::string modelsDir, modelDetPath, modelClsPath, modelRecPath, keysPath;
    std::string imgPath, imgDir, imgName;
    int numThread = 4;
    int padding = 50;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.6f;
    float boxThresh = 0.3f;
    float unClipRatio = 2.0f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = true;
    int flagMostAngle = 1;

    OcrLite ocrLite;

    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
        true,//isOutputConsole
        false,//isOutputPartImg
        true);//isOutputResultImg


    ocrLite.enableResultTxt("resimg", "res.jpg");

    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
        "numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d)\n",
        numThread, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    modelsDir = "D:\\tool_develop\\chineseOCR_lite\\Onnx_OCR\\Onnx_OCR\\x64\\Release\\models\\";
    //"D:\\tool_develop\\chineseOCR_lite\\rapidocr-win-all\\x64\\onnx-models\\";
    modelDetPath = modelsDir + "ch_ppocr_mobile_v2.0_det_infer.onnx";//"dbnet.onnx";
    modelClsPath = modelsDir + "model.onnx";//"model_class.onnx";
    modelRecPath = modelsDir + "ch_PP-OCRv2_rec_infer.onnx";
    keysPath = modelsDir + "ppocr_keys_v1.txt";

    imgDir = "D:\\tool_develop\\chineseOCR_lite\\OcrLiteOnnx-1.6.1\\images\\";
    imgName = "1_1.jpg";

    ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);
    for (size_t i = 0; i < 20; i++)
    {
        OcrResult result = ocrLite.detect(imgDir.c_str(), imgName.c_str(), padding, maxSideLen,
            boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
        ocrLite.Logger("%s\n", result.strRes.c_str());
    }

    //OcrResult result = ocrLite.detect(imgDir.c_str(), imgName.c_str(), padding, maxSideLen,
    //    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);



    std::cout << "Hello World!\n";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
