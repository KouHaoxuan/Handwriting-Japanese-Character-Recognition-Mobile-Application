package com.example.dg0

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DigitClassifier(private val context:Context) {
    private var interpreter: Interpreter? = null
    var isInitialized = false
    //private  set

    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0
    private var inputImageHeight:Int = 0
    private var modelInputSize: Int = 0

    fun initialize():Task<Void>{
        val task = TaskCompletionSource<Void>()
        executorService.execute(){
            try {
                initializeInterpreter()
                task.setResult(null)
            }catch (e:IOException){
                task.setException(e)
            }
        }
        return task.task
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename:String):ByteBuffer{
        //对assets目录的资源文件进行访问
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        //使用了java.nio的channel模式
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    private fun initializeInterpreter(){
        val assetManager = context.assets
        val model = loadModelFile(assetManager, "mnist1.tflite")
        val interpreter = Interpreter(model)

        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE

        this.interpreter = interpreter

        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")

    }

    private fun classify(bitmap: Bitmap): String {
        check(isInitialized){"TF Lite Interpreter is not initialized yet."}

        val resizedImage = Bitmap.createScaledBitmap(
            bitmap,
            inputImageWidth,
            inputImageHeight,
            true
        )

        val byteBuffer = convertBitmapToByteBuffer(resizedImage)

        val output = Array(1){FloatArray(OUTPUT_CLASSES_COUNT)}
        interpreter?.run(byteBuffer, output)

        val result = output[0]
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        val hiragana = HIRAGANA_MAP[maxIndex] ?: "不明"
        val resultString =
            "予測結果: %s (%d)\n確信度: %.2f".format(hiragana, maxIndex, result[maxIndex])
        return resultString
    }

    fun classifyAsync(bitmap: Bitmap):Task<String>{
        val task = TaskCompletionSource<String>()
        executorService.execute(){
            val result = classify(bitmap)
            task.setResult(result)
        }
        return task.task
    }

    fun close(){
        executorService.execute(){
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap):ByteBuffer{
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for(pixelValue in pixels){
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            val normalizedPixelValue = (r + g + b)/3.0f/255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }

        return byteBuffer
    }

    companion object {
        private const val TAG = "DigitClassifier"
        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 1
        private const val OUTPUT_CLASSES_COUNT = 49

        private val HIRAGANA_MAP = mapOf(
            0 to "あ", 1 to "い", 2 to "う", 3 to "え", 4 to "お",
            5 to "か", 6 to "き", 7 to "く", 8 to "け", 9 to "こ",
            10 to "さ", 11 to "し", 12 to "す", 13 to "せ", 14 to "そ",
            15 to "た", 16 to "ち", 17 to "つ", 18 to "て", 19 to "と",
            20 to "な", 21 to "に", 22 to "ぬ", 23 to "ね", 24 to "の",
            25 to "は", 26 to "ひ", 27 to "ふ", 28 to "へ", 29 to "ほ",
            30 to "ま", 31 to "み", 32 to "む", 33 to "め", 34 to "も",
            35 to "や", 36 to "ゆ", 37 to "よ",
            38 to "ら", 39 to "り", 40 to "る", 41 to "れ", 42 to "ろ",
            43 to "わ", 44 to "ゐ", 45 to "ゑ", 46 to "を", 47 to "ん",
            48 to "ゝ"
        )
    }
}