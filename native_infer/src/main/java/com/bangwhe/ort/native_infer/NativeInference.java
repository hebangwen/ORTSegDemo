package com.bangwhe.ort.native_infer;


import java.io.File;
import java.io.FileNotFoundException;

public class NativeInference {

    static {
        System.loadLibrary("native_infer");
    }

    public static long dummyInference(String onnxPath, int warmupIters, int inferenceIters) throws FileNotFoundException {
        File file = new File(onnxPath);
        if (!file.exists()) {
            throw new FileNotFoundException(String.format("%s not found!!!", onnxPath));
        }

        return nativeDummyInference(onnxPath, warmupIters, inferenceIters);
    }

    private static native long nativeDummyInference(String onnxPath, int warmupIters, int inferenceIters);
}