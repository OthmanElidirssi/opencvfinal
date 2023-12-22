package com.example.opencvfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import android.os.Bundle;

import android.util.Log;
import android.view.MotionEvent;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    private ImageView image;
    private List<Point> clickedPoints = new ArrayList<>();

    private TextView left;

    private TextView right;


    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    //your code
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        image = findViewById(R.id.dent_image);
        left=findViewById(R.id.left);
        right=findViewById(R.id.right);

        // Initialize OpenCV asynchronously
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mOpenCVCallBack);
        } else {
            Log.d("OpenCV", "OpenCV initialization succeeded.");
            // OpenCV is already initialized, you can execute your code here
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.dent1);
            processImageAndDrawLines(bitmap);
        }
    }

    private void processImageAndDrawLines(Bitmap originalBitmap) {
        // Convert Bitmap to Mat with CV_8UC4 type
        Mat matImage = new Mat();
        Utils.bitmapToMat(originalBitmap, matImage);

        // Perform image processing
        Mat edges = preprocessImage(matImage);

        // Find and draw points
        MatOfPoint points = findAndDrawPoints(edges, matImage);

        // Display the processed image with points in ImageView
        Bitmap processedBitmap = Bitmap.createBitmap(originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matImage, processedBitmap);
        image.setImageBitmap(processedBitmap);

        // Set click listener for further interaction
        image.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN && clickedPoints.size() < 4) {
                int x = (int) event.getX();
                int y = (int) event.getY();
                Mat processedImage = mouseClick(points, x, y, matImage);

                Mat linesImage = drawLinesBetweenPoints(processedImage);

                // Convert Mat to Bitmap with CV_8UC4 type
                Bitmap linesBitmap = Bitmap.createBitmap(originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(linesImage, linesBitmap);
                image.setImageBitmap(linesBitmap);
            } else if (clickedPoints.size()==4) {
                calculateAngles(clickedPoints);
            }
            return true;
        });
    }


    private Mat preprocessImage(Mat image) {
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 50, 150);
        return edges;
    }

    private MatOfPoint findAndDrawPoints(Mat edges, Mat originalImage) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat processedImage = Mat.zeros(edges.size(), CvType.CV_8UC3);
        List<Point> pointsArray = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            for (int i = 0; i < contour.rows(); i++) {
                double[] point = contour.get(i, 0);
                double x = point[0];
                double y = point[1];
                pointsArray.add(new Point(x, y));
                Imgproc.circle(originalImage, new Point(x, y), 2, new Scalar(255, 0, 0), -1);
            }
        }

        return new MatOfPoint(pointsArray.toArray(new Point[0]));
    }

    private Mat mouseClick(MatOfPoint points, int x, int y, Mat originalImage) {
        double[] distances = new double[points.rows()];
        for (int i = 0; i < points.rows(); i++) {
            Point point = points.toList().get(i);
            distances[i] = Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2));
        }

        int closestPointIndex = findIndexOfMinValue(distances);
        Point closestPoint=points.toList().get(closestPointIndex);
        Mat processedImage = new Mat();
        originalImage.copyTo(processedImage);

        for (Point point : clickedPoints) {
            Imgproc.circle(processedImage, point, 10, new Scalar(255, 0, 0), -1);
        }

        clickedPoints.add(closestPoint);

        Imgproc.circle(processedImage, closestPoint, 10, new Scalar(255, 0, 0), -1);

        return processedImage;
    }

    private int findIndexOfMinValue(double[] array) {
        int minIndex = 0;
        double minValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < minValue) {
                minValue = array[i];
                minIndex = i;
            }
        }
        return minIndex;
    }
    private Mat drawLinesBetweenPoints(Mat image) {
        if (clickedPoints.size() == 4) {
            Mat processedImage = new Mat();
            image.copyTo(processedImage);

            // Draw lines between the selected points
            Imgproc.line(processedImage, clickedPoints.get(0), clickedPoints.get(1), new Scalar(255, 255, 0), 3);
            Imgproc.line(processedImage, clickedPoints.get(2), clickedPoints.get(3), new Scalar(255, 255, 0), 3);

            return processedImage;
        } else {
            return image;
        }
    }


    private void calculateAngles(List<Point> points) {
        if (points.size() == 4) {
            Point p1 = points.get(0);
            Point p2 = points.get(1);
            Point p3 = points.get(2);
            Point p4 = points.get(3);
            double angle2 = Math.toDegrees(Math.atan((p2.y - p1.y) / (p2.x - p1.x)));
            double angle3 = Math.toDegrees(Math.atan((p4.y - p3.y) / (p4.x - p3.x)));
            setUI(angle2,angle3);
            Log.d("Angle", "Deviation angle for the first line: " + angle2);
            Log.d("Angle", "Deviation angle for the second line: " + angle3);
        }
    }




    private void setUI(double left,double right){
        this.left.setText(String.valueOf(left));
        this.right.setText(String.valueOf(right));
    }



}