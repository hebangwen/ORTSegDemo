package com.bangwhe.ort.javasegdemo;

import android.graphics.Bitmap;
import android.util.Log;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;

public class ORTAnalyzer {
    OrtSession session;
    String TAG = "ORTAnalyzer";
    int inferenceIterations = 20;
    int warmupIterations = 5;

    public ORTAnalyzer(OrtSession session) {
        this.session = session;
    }

    public void dummyAnalyze() throws OrtException {
        OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        try {
            Map<String, NodeInfo> inputInfoMap = session.getInputInfo();
            for (Map.Entry<String, NodeInfo> entry : inputInfoMap.entrySet()) {
                String inputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                Log.d(TAG, String.format("InputName: %s, JavaType: %s, ONNXType: %s, Shape: %s",
                        inputName, tensorInfo.type, tensorInfo.onnxType, Arrays.toString(tensorInfo.getShape())));

                long[] shape = tensorInfo.getShape();
                Object dummyArray = OrtUtil.newFloatArray(shape);

                OnnxTensor onnxTensor =OnnxTensor.createTensor(ortEnvironment, dummyArray);
                inputMap.put(inputName, onnxTensor);
            }

            Map<String, NodeInfo> outputInfoMap = session.getOutputInfo();
            for (Map.Entry<String, NodeInfo> entry : outputInfoMap.entrySet()) {
                String outputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                Log.d(TAG, String.format("OutputName: %s, JavaType: %s, ONNXType: %s, Shape: %s",
                        outputName, tensorInfo.type, tensorInfo.onnxType, Arrays.toString(tensorInfo.getShape())));
            }

            for (int i = 0; i < warmupIterations; i++) {
                long startTime = System.currentTimeMillis();
                session.run(inputMap);
                long endTime = System.currentTimeMillis();
                Log.d(TAG, String.format("Warmup time cost: %d ms", endTime - startTime));
            }

            long inferenceStartTime = System.currentTimeMillis();
            for (int i = 0; i < inferenceIterations; i++) {
                long startTime = System.currentTimeMillis();
                session.run(inputMap);
                long endTime = System.currentTimeMillis();
                Log.d(TAG, String.format("Time cost: %d ms", endTime - startTime));
            }
            long inferenceEndTime = System.currentTimeMillis();
            Log.d(TAG, String.format("Average time cost: %d ms", (inferenceEndTime - inferenceStartTime) / inferenceIterations));
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public void analyze(Bitmap bitmap) throws OrtException {
        OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        try {
            Map<String, NodeInfo> inputInfoMap = session.getInputInfo();
            for (Map.Entry<String, NodeInfo> entry : inputInfoMap.entrySet()) {
                String inputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                long[] shape = tensorInfo.getShape();

                Log.d(TAG, String.format("InputName: %s, JavaType: %s, ONNXType: %s, Shape: %s",
                        inputName, tensorInfo.type, tensorInfo.onnxType, Arrays.toString(shape)));

                FloatBuffer inputArray = preprocess(bitmap);
                OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, inputArray, shape);
                inputMap.put(inputName, onnxTensor);
            }

            Map<String, NodeInfo> outputInfoMap = session.getOutputInfo();
            for (Map.Entry<String, NodeInfo> entry : outputInfoMap.entrySet()) {
                String outputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
                Log.d(TAG, String.format("OutputName: %s, JavaType: %s, ONNXType: %s, Shape: %s",
                        outputName, tensorInfo.type, tensorInfo.onnxType, Arrays.toString(tensorInfo.getShape())));
            }

            for (int i = 0; i < warmupIterations; i++) {
                long startTime = System.currentTimeMillis();
                session.run(inputMap);
                long endTime = System.currentTimeMillis();
                Log.d(TAG, String.format("Warmup time cost: %d ms", endTime - startTime));
            }

            long inferenceStartTime = System.currentTimeMillis();
            for (int i = 0; i < inferenceIterations; i++) {
                long startTime = System.currentTimeMillis();
                session.run(inputMap);
                long endTime = System.currentTimeMillis();
                Log.d(TAG, String.format("Time cost: %d ms", endTime - startTime));
            }
            long inferenceEndTime = System.currentTimeMillis();
            Log.d(TAG, String.format("Average time cost: %d ms", (inferenceEndTime - inferenceStartTime) / inferenceIterations));
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    final int DIM_BATCH_SIZE = 1;
    final int DIM_PIXEL_SIZE = 3;
    final int IMAGE_SIZE_X = 384;
    final int IMAGE_SIZE_Y = 672;
    private FloatBuffer preprocess(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        FloatBuffer imgData = FloatBuffer.allocate(DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_X * IMAGE_SIZE_Y);
        imgData.rewind();

        int stride = IMAGE_SIZE_X * IMAGE_SIZE_Y;
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, width, 0, 0, width, height);
        for (int i = 0; i < IMAGE_SIZE_X; i++) {
            for (int j = 0; j < IMAGE_SIZE_Y; j++) {
                int idx = IMAGE_SIZE_Y * i + j;
                int pixelValue = bmpData[idx];
                imgData.put(idx, ((pixelValue >> 16 & 0xFF) / 255f - 0.485f) / 0.229f);
                imgData.put(idx + stride, ((pixelValue >> 8 & 0xFF) / 255f - 0.456f) / 0.224f);
                imgData.put(idx + stride * 2, ((pixelValue & 0xFF) / 255f - 0.406f) / 0.225f);
            }
        }

        imgData.rewind();
        return imgData;
    }
}
