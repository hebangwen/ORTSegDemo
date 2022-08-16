package com.bangwhe.ort.native_infer;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private TextView inferenceView;
    String mobilnetv2Path = "/data/local/tmp/mobilenetv2_7.onnx";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inferenceView = findViewById(R.id.inference);

        String filepath = mobilnetv2Path;
        try {
            long latency = NativeInference.dummyInference(filepath, 5, 30);
            inferenceView.setText(String.format("inference time: %dms", latency));
            Log.d(TAG, String.format("Average time cost: %dms", latency));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            inferenceView.setText(String.format("%s not found!!!", filepath));
        }
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