package com.bangwhe.ort.javasegdemo;

import android.graphics.Bitmap;
import android.util.Log;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

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
        Set<String> inputNamesSet = session.getInputNames();
        String[] inputNames = new String[inputNamesSet.size()];
        inputNamesSet.toArray(inputNames);

        Random random = new Random();

        String inputName1 = inputNames[0];
        long[] shape1 = new long[] {1, 128, 2};
        // 生成 float[][][] 三维数组
//        Object dummyArray1 =  OrtUtil.newFloatArray(shape1);
        long totalLength1 = shape1[0] * shape1[1] * shape1[2];
        FloatBuffer dummyInput1 = FloatBuffer.allocate((int) totalLength1);
        for (int i = 0; i < totalLength1; i++) {
            dummyInput1.put(random.nextFloat());
        }
        Log.d(TAG, inputName1);

        String inputName2 = inputNames[1];
        long[] shape2 = new long[] {1, 64, 96, 128};
        long totalLength2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];
        FloatBuffer dummyInput2 = FloatBuffer.allocate((int) totalLength2);
        for (int i = 0; i < totalLength2; i++) {
            dummyInput2.put(random.nextFloat());
        }
        Log.d(TAG, inputName2);
        Log.d(TAG, totalLength1 + " " + totalLength2);

        dummyInput1.rewind();
        dummyInput2.rewind();

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        try {
            inputMap.put(inputName1, OnnxTensor.createTensor(ortEnvironment, dummyInput1, shape1));
            inputMap.put(inputName2, OnnxTensor.createTensor(ortEnvironment, dummyInput2, shape2));

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
