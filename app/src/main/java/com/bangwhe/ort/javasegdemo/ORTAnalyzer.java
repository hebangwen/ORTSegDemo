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

    public void dummyAnalyze() {
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
}
