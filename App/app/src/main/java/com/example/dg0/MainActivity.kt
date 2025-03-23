package com.example.dg0

import android.annotation.SuppressLint
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
//import android.widget.Toast
import com.divyanshu.draw.widget.DrawView
//import com.example.dg0.DigitClassifier

class MainActivity : AppCompatActivity() {
    private var drawView: DrawView? = null
    private var clearButton: Button? = null
    private var predictedTextView: TextView? = null
    private val digitClassifier = DigitClassifier(this)

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        drawView = findViewById(R.id.draw_view)
        drawView?.setStrokeWidth(70.0f)
        drawView?.setColor(Color.WHITE)
        drawView?.setBackgroundColor(Color.BLACK)

        clearButton = findViewById(R.id.clear_button)
        predictedTextView = findViewById(R.id.predicted_text)

        clearButton?.setOnClickListener() {
            drawView?.clearCanvas()
            predictedTextView?.text = getString(R.string.prediction_text_placeholder)
        }

        drawView?.setOnTouchListener() { _, event ->
            drawView?.onTouchEvent(event)

            if (event.action == MotionEvent.ACTION_UP) {
                //Toast.makeText(this, "ACTION UP", Toast.LENGTH_SHORT).show()
                classifyDrawing()
            }
            true

        }

        digitClassifier.initialize().addOnFailureListener(){
            e -> Log.e(TAG, "Error to setting up digit classifier.",e)
        }

    }

    override fun onDestroy() {
        digitClassifier.close()
        super.onDestroy()
    }

    private fun classifyDrawing(){
        val bitmap = drawView?.getBitmap()

        if((bitmap != null) && (digitClassifier.isInitialized)){
            digitClassifier
                .classifyAsync(bitmap)
                .addOnSuccessListener() {resultText -> predictedTextView?.text = resultText }
                .addOnFailureListener(){e ->
                    predictedTextView?.text = e.localizedMessage.toString()
//                    predictedTextView?.text = getString(R.string.classification_error_message,
//                    e.localizedMessage)
                    Log.e(TAG, "Error classifying drawing.", e)
                }
        }
    }



    companion object{
        private const val TAG = "MainActivity"
    }

}

