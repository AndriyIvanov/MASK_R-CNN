//*****************************************************************************//
//��������� ����������� ����� �� ����������� �� ������ ��������� "Mask R-CNN"  // 
//    (https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN)		   //		
//*****************************************************************************//
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <ctime>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// ������������� ����������
float confThreshold = 0.5;					    // ����������� ������������� �����������
float maskThreshold = 0.4;					    // ����������� ����� (���������� ����� ��� ���������� ����� � ������� 0)
string inputName = "Resources/person_01.bmp";	// �������� ����� ����������� ��� ������� 

vector<string> classes;

void segmentationObject(Mat& inputImage, Net& net, vector<Mat>& outputMasks, Rect& outputRect);

int main(int argc, char** argv)
{
	// �������� ������������ �������
	cout << "Parameters initialisation ..." << endl;
	string classesFile = "Parameters/mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// ����������� ��������� � ����� ������
	String textGraph = "Parameters/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String modelWeights = "Parameters/frozen_inference_graph.pb";

	// �������� ����
	cout << "Network downloading ..." << endl;
	Net net = readNetFromTensorflow(modelWeights, textGraph);

	// ��������� ������ �� CPU
	//net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(DNN_TARGET_CPU);
	// ��������� ������ �� GPU
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);

	Mat inputImage = imread(inputName);
	if (inputImage.empty())
	{
		cout << "Can't open the file!" << endl;
		cin.get();
		return 0;
	}
	
	vector<Mat> masks;
	Rect outputRect;
	cout << "Image segmentation ..." << endl;
	clock_t tStart = clock();
	segmentationObject(inputImage, net, masks, outputRect);
	cout << "Segmentation time: " << (clock() - tStart) / (double)CLOCKS_PER_SEC << endl;

	//��������� �����������
	imshow("Detected objects", inputImage);

	//��������� �����
	for (int i=0; i<masks.size(); i++)
	{
		string Mask = "Mask ";
		Mask += ('0' + i + 1);
		imshow(Mask, masks[i]);
	}
	waitKey(0);
	return 0;
}

void segmentationObject(Mat& inputImage, Net& net, vector<Mat>& outputMasks, Rect& outputRect)
{
	// �������� 4D-����� �� �������� �����������
	Mat blob;
	blobFromImage(inputImage, blob, 1.0, Size(inputImage.cols, inputImage.rows), Scalar(), true, false);
	// ��������� ����� ����
	net.setInput(blob);
	// ������ ������ ����
	std::vector<String> outNames(2);
	outNames[0] = "detection_out_final";
	outNames[1] = "detection_masks";
	vector<Mat> outs;
	net.forward(outs, outNames);

	// ���������� ����� � �����
	// �������� ������ ����� NxCxHxW, ���
	// N - ���������� ��������� ��������
	// C - ���������� �������
	// HxW - ����� �������
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	const int numDetections = outDetections.size[2];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int classId, left, top, right, bottom, i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			classId = static_cast<int>(outDetections.at<float>(i, 1));
			if (classId < (int)classes.size())
			{
				left = static_cast<int>(inputImage.cols * outDetections.at<float>(i, 3));
				top = static_cast<int>(inputImage.rows * outDetections.at<float>(i, 4));
				right = static_cast<int>(inputImage.cols * outDetections.at<float>(i, 5));
				bottom = static_cast<int>(inputImage.rows * outDetections.at<float>(i, 6));

				left = max(0, min(left, inputImage.cols - 1));
				top = max(0, min(top, inputImage.rows - 1));
				right = max(0, min(right, inputImage.cols - 1));
				bottom = max(0, min(bottom, inputImage.rows - 1));
				outputRect = Rect(left, top, right - left + 1, bottom - top + 1);

				// ���������� ����� �������
				Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
				// ���������� ����� �� �����������
				rectangle(inputImage, outputRect.tl(), outputRect.br(), Scalar(255, 255, 255), 2);
				//�������������� ����� � ��������� �������
				resize(objectMask, objectMask, Size(outputRect.width, outputRect.height));
				Mat mask = (objectMask > maskThreshold);
				mask.convertTo(mask, CV_8U);
				outputMasks.push_back(mask);
			}
		}
	}
}
