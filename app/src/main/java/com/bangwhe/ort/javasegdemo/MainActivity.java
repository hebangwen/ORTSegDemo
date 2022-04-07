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
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private OrtEnvironment mOrtEnvironment;
    private OrtSession.SessionOptions mSessionOptions;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOrtEnvironment = OrtEnvironment.getEnvironment();
        mSessionOptions = new OrtSession.SessionOptions();
        setORTAnalyzer();
    }

    private void setORTAnalyzer() {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.frame_980_672x384);
        ORTAnalyzer ortAnalyzer = new ORTAnalyzer(createSession());

        try {
            ortAnalyzer.analyze(bitmap);
        } catch (NullPointerException e) {
            e.printStackTrace();
        }
    }

    private OrtSession createSession() {
        try {
//            mSessionOptions.addNnapi();
            mSessionOptions.setIntraOpNumThreads(4);
//            mSessionOptions.addNnapi(EnumSet.of(NNAPIFlags.USE_NCHW));
//            mSessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

            InputStream inputStream = getResources().openRawResource(R.raw.gcn_with_runtime_opt);
            byte[] bytes = readBytes(inputStream);
            return mOrtEnvironment.createSession(bytes, mSessionOptions);
        } catch (OrtException | IOException e) {
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