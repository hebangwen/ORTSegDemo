package com.bangwhe.ort.javasegdemo;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.renderscript.ScriptGroup;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.EnumSet;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private OrtEnvironment mOrtEnvironment;
    private OrtSession.SessionOptions mSessionOptions;

    int snakeResId = R.raw.gcn_with_runtime_opt;
    int segFormerB0ResId = R.raw.segformer_b0_1024x1024;
    int segFormerB0ONNXResId = R.raw.segformer_b0_1024x1024_onnx;
    int mobilenetONNXResId = R.raw.mobilenetv2_7;
    String segFormerB5Path = "/data/local/tmp/segformer_b5_1024x1024.with_runtime_opt.ort";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
        for (OrtProvider provider : providers) {
            Log.d(TAG, "ONNXRuntime available provider: " + provider);
        }

        mOrtEnvironment = OrtEnvironment.getEnvironment();
        mSessionOptions = new OrtSession.SessionOptions();
        setORTAnalyzer();
    }

    private void setORTAnalyzer() {
        ORTAnalyzer ortAnalyzer = new ORTAnalyzer(createSession(mobilenetONNXResId));

        try {
            ortAnalyzer.dummyAnalyze();
        } catch (NullPointerException e) {
            e.printStackTrace();
        }
    }

    private void configSessionOptions() throws OrtException {
//        mSessionOptions.addNnapi();  // use Android NNAPI, thus inference in GPU
        mSessionOptions.setIntraOpNumThreads(4);  // set number of threads in CPU
    }

    private OrtSession createSession(int resId) {
        try {
            configSessionOptions();

            InputStream inputStream = getResources().openRawResource(resId);
            byte[] bytes = readBytes(inputStream);
            return mOrtEnvironment.createSession(bytes, mSessionOptions);
        } catch (OrtException | IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    private OrtSession createSession(String modelPath) {
        try {
            configSessionOptions();
            return mOrtEnvironment.createSession(modelPath, mSessionOptions);
        } catch (OrtException e) {
            e.printStackTrace();
        }

        return null;
    }

    public byte[] readBytes(InputStream inputStream) throws IOException {
        // this dynamically extends to take the bytes you read
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

        // this is storage overwritten on each iteration with bytes
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        // we need to know how may bytes were read to write them to the byteBuffer
        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }

        // and then we can return your byte array.
        return byteBuffer.toByteArray();
    }
}