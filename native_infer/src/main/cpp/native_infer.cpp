//
// Created by Bangwen on 2022/8/12.
//

#include <iostream>
#include <jni.h>
#include <string>
#include <chrono>
#include <sstream>
#include <android/log.h>

#include "onnxruntime_c_api.h"
#include "experimental_onnxruntime_cxx_api.h"

#define TAG "NATIVE_INFERENCE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte* pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *)pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

// pretty prints a shape dimension vector
template<typename T>
std::string print_shape(const std::vector<T>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

template<typename T>
int calculate_product(const std::vector<T>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

long dummyInference(std::string onnx_filepath, int warmup_iters, int inference_iters) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nanodet_plus");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);
    Ort::Experimental::Session session(env, onnx_filepath, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // print name/shape of inputs
    auto input_names = session.GetInputNames();
    auto input_shapes = session.GetInputShapes();
    auto inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::stringstream ss("");
    ss << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names.size(); i++) {
        ss << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
    }

    // print name/shape of outputs
    auto output_names = session.GetOutputNames();
    auto output_shapes = session.GetOutputShapes();
    ss << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names.size(); i++) {
        ss << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
    }
    LOGD("%s", ss.str().c_str());
    ss.str("");

    // generate random inputs
    std::vector<Ort::Value> input_tensors;
    for (auto data_shape: input_shapes) {
        int total_number_elements = calculate_product(data_shape);
        std::vector<float> data_values(total_number_elements);
        std::generate(data_values.begin(), data_values.end(), [&] {return rand() % 255; });
        input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
                data_values.data(), data_values.size(), data_shape));
    }

    long latency = 0;
    try {
        for (int i = 0; i < warmup_iters; i++) {
            auto start = std::chrono::system_clock::now();
            auto output_tensors = session.Run(
                    session.GetInputNames(),
                    input_tensors,
                    session.GetOutputNames());
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start).count();
            ss << "Warmup time cost: " << duration << " ms" << std::endl;
            LOGD("%s", ss.str().c_str());
            ss.str("");
            assert(output_tensors.size() == session.GetOutputNames().size() &&
                   output_tensors[0].IsTensor());
            for (int j = 0; j < output_tensors.capacity(); j++) {
                output_tensors[j].release();
            }
        }

        for (int i = 0; i < inference_iters; i++) {
            auto start = std::chrono::system_clock::now();
            auto output_tensors = session.Run(
                    session.GetInputNames(),
                    input_tensors,
                    session.GetOutputNames());
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            ss << "Time cost: " << duration << " ms" << std::endl;
            LOGD("%s", ss.str().c_str());
            ss.str("");
            assert(output_tensors.size() == session.GetOutputNames().size() &&
                   output_tensors[0].IsTensor());
            latency += duration;
            for (int j = 0; j < output_tensors.capacity(); j++) {
                output_tensors[j].release();
            }
        }

        latency /= inference_iters;
    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }

    return latency;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_bangwhe_ort_native_1infer_NativeInference_nativeDummyInference(JNIEnv *env, jclass thiz,
                                                                  jstring onnx_path,
                                                                  jint warmup_iters,
                                                                  jint inference_iters) {

    std::string c_onnx_filepath = jstring2string(env, onnx_path);
    long latency = dummyInference(c_onnx_filepath, warmup_iters, inference_iters);
    return latency;
}