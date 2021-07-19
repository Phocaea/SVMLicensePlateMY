#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "dirent.h"
#include <string>


using namespace cv;
using namespace std;
using namespace cv::ml;

const int number_of_feature = 32;

static int count_pixel(Mat img, bool black_pixel = true)
{
	int black = 0;
	int white = 0;
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (img.at<uchar>(i, j) == 0)
				black++;
			else
				white++;
		}
	if (black_pixel)
		return black;
	else
		return white;
}

static vector<float> calculate_feature(Mat src)
{
	Mat img;
	if (src.channels() == 3)
	{
		cvtColor(src, img, COLOR_BGR2GRAY);
		threshold(img, img, 100, 255, THRESH_BINARY);
	}
	else
	{
		threshold(src, img, 100, 255, THRESH_BINARY);
	}

	vector<float> r;
	//vector<int> cell_pixel;
	resize(img, img, Size(40, 40));
	int h = img.rows / 4;
	int w = img.cols / 4;
	int S = count_pixel(img);
	int T = img.cols * img.rows;
	for (int i = 0; i < img.rows; i += h)
	{
		for (int j = 0; j < img.cols; j += w)
		{
			Mat cell = img(Rect(i, j, h, w));
			int s = count_pixel(cell);
			float f = (float)s / S;
			r.push_back(f);
		}
	}

	for (int i = 0; i < 16; i += 4)
	{
		float f = r[i] + r[i + 1] + r[i + 2] + r[i + 3];
		r.push_back(f);
	}

	for (int i = 0; i < 4; ++i)
	{
		float f = r[i] + r[i + 4] + r[i + 8] + r[i + 12];
		r.push_back(f);
	}

	r.push_back(r[0] + r[5] + r[10] + r[15]);
	r.push_back(r[3] + r[6] + r[9] + r[12]);
	r.push_back(r[0] + r[1] + r[4] + r[5]);
	r.push_back(r[2] + r[3] + r[6] + r[7]);
	r.push_back(r[8] + r[9] + r[12] + r[13]);
	r.push_back(r[10] + r[11] + r[14] + r[15]);
	r.push_back(r[5] + r[6] + r[9] + r[10]);
	r.push_back(r[0] + r[1] + r[2] + r[3] + r[4] + r[7] + r[8] + r[11] + r[12] + r[13] + r[14] + r[15]);

	return r; //32 feature
}

char character_recognition(Mat img_character)
{
	//Load SVM training file OpenCV 3.1
	Ptr<SVM> svmNew = SVM::create();
	svmNew = SVM::load("svm.txt");
	char c = '*';

	vector<float> feature = calculate_feature(img_character);
	// Open CV3.1
	Mat m = Mat(1, number_of_feature, CV_32FC1);
	for (size_t i = 0; i < feature.size(); ++i)
	{
		float temp = feature[i];
		m.at<float>(0, i) = temp;
	}

	int ri = int(svmNew->predict(m)); // Open CV 3.1
									  /*int ri = int(svmNew.predict(m));*/
	if (ri >= 0 && ri <= 9)
		c = (char)(ri + 48); //ma ascii 0 = 48
	if (ri >= 10 && ri < 18)
		c = (char)(ri + 55); //ma accii A = 5, --> tu A-H
	if (ri >= 18 && ri < 22)
		c = (char)(ri + 55 + 2); //K-N, bo I,J
	if (ri == 22) c = 'P';
	if (ri == 23) c = 'S';
	if (ri >= 24 && ri < 27)
		c = (char)(ri + 60); //T-V,  
	if (ri >= 27 && ri < 30)
		c = (char)(ri + 61); //X-Z

	return c;

}

bool trainSVM(string savepath, string trainPath)
{
	const int number_of_class = 30;
	const int number_of_sample = 10;
	const int number_of_feature = 32;

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(0.5);
	svm->setC(16);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	vector<string> img_path;
	DIR* dir = opendir(trainPath.c_str());
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL)
	{
		if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0))
		{
			string folder_path = trainPath + "/" + string(entry->d_name);
			img_path.push_back(folder_path);
		}
	}
	closedir(dir);
	if (img_path.size() <= 0)
	{
		//do something to check whether path is found and etc
		return false;
	}
	if (number_of_class != img_path.size() || number_of_sample <= 0 || number_of_class <= 0)
	{
		//do something to check whether path is found and etc
		return false;
	}
	Mat data = Mat(number_of_class * number_of_sample, number_of_feature, CV_32FC1);
	Mat label = Mat(number_of_class * number_of_sample, 1, CV_32SC1);

	int index = 0;
	sort(img_path.begin(), img_path.end());
	for (size_t i = 0; i < img_path.size(); ++i)
	{
		vector<string> files;
		DIR* dir = opendir(img_path.at(i).c_str());
		struct dirent *entry1;
		while ((entry1 = readdir(dir)) != NULL)
		{
			if ((strcmp(entry1->d_name, ".") != 0) && (strcmp(entry1->d_name, "..") != 0))
			{
				string folder_path = img_path.at(i) + "/" + string(entry1->d_name);
				files.push_back(folder_path);
			}
		}
		closedir(dir);
		if (files.size() <= 0 || files.size() != number_of_sample)
		{
			return false;
		}
		string file_path1 = img_path.at(i);
		cout << "list folder" << img_path.at(i) << endl;
		string label_folder = file_path1.substr(file_path1.length() - 1);
		for (size_t j = 0; j < files.size(); ++j)
		{
			Mat src = imread(files.at(j));
			//bitwise_not(src, src);
			if (src.empty())
			{
				return false;
			}
			vector<float>feature = calculate_feature(src);
			for (size_t k = 0; k < feature.size(); ++k)
				data.at<float>(index, k) = feature.at(k);
			label.at<int>(index, 0) = i;
			index++;
		}
	}

	svm->trainAuto(TrainData::create(data, ROW_SAMPLE, label));
	svm->save(savepath);
	return true;
}

int main()
{
	string licenseNumber;
	vector<Mat> plates;
	vector<Mat> draw_char;
	vector<vector<Mat>>characters;
	vector<string> text;
	vector<Mat> charc;

	Mat image = imread("D:\\CarplateDS\\Dataset\\JLP911 26.jpg");
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat equalize;
	equalizeHist(gray, equalize);
	Mat blurImg;
	blur(equalize, equalize, Size(3, 3));
	GaussianBlur(equalize, blurImg,Size(3, 3), 2);
	Mat edge;
	//Mat binary;
	adaptiveThreshold(blurImg, edge, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 33, 5);
	//Canny(blur, edge, 30, 200);
	//imshow("Edge", edge);
	//waitKey();
	
	Mat Opening;
	Mat kernel = getStructuringElement(MORPH_RECT,Size(3, 3));
	morphologyEx(edge, Opening,MORPH_OPEN, kernel,Point(-1,-1),1);
	//imshow("img1", Opening);
	//waitKey();

	Mat Blob = Opening.clone();
	vector<vector<Point>>contours1;
	//vector<Vec4i>hierarchy1;

	findContours(edge,contours1, RETR_TREE, CHAIN_APPROX_NONE);

	Mat dst = Mat::zeros(gray.size(), CV_8UC1);
	Mat mask = Mat::zeros(image.size(), CV_8UC1);	
	Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));

	//draw the contours for segment into different parts
	if (!contours1.empty())
	{
		for (int i = 0; i < contours1.size() ; i++)
		{
			drawContours(dst, contours1, i, colour,2, FILLED);
		}
	}

	Rect BlobRect;
	Scalar colour1(0, 255,0);
	Rect roi;
	Mat Cropped;
	Mat sub_image = image.clone();
	Mat sub_binary;

	for (int j = 0; j < contours1.size(); j++)
	{
		BlobRect = boundingRect(contours1[j]);
		if(BlobRect.width < 40 || BlobRect.width >150 || BlobRect.height > 180  || BlobRect.area() < 1200
			|| BlobRect.height < 20 || (BlobRect.width / BlobRect.height) < 2
			|| BlobRect.x < 300 || BlobRect.y < 300 || BlobRect.x > 600 || BlobRect.y > 500 
			/*||(contourArea(contours1[j]) / (double)(BlobRect.width * BlobRect.height)) < 0.1 || BlobRect.x > (Blob.rows -100)*/)
		{
			drawContours(Blob, contours1, j, colour1,FILLED);
		}		
		else
		{
			//roi = Rect(BlobRect.x, BlobRect.y, BlobRect.width, BlobRect.height);
			rectangle(sub_image, BlobRect, Scalar(255,0,0),2);
			Cropped = image(BlobRect);
		}
	}

	 
	//imshow("Blob", sub_image);
	//waitKey();	

	Mat CroppedGray;
	Mat CropBinary;
	Mat CropOpen;
	Rect CroppedRect;
	vector<vector<Point>>contours2;
	cvtColor(Cropped, CroppedGray, COLOR_BGR2GRAY);
	adaptiveThreshold(CroppedGray, CropBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 3);
	/*Depends on situation for morphological process*/
	morphologyEx(CropBinary, CropOpen, MORPH_ERODE, kernel, Point(-1, -1), 1.5);
	findContours(CropOpen, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<Rect> r_char;
	for (int i = 0; i < contours2.size(); i++)
	{
		CroppedRect = boundingRect(contours2[i]);
		if (CroppedRect.height < 30 && CroppedRect.width < 15 && CroppedRect.x < 110 && CroppedRect.x > 10 && CroppedRect.area() > 20)
		{
			r_char.push_back(CroppedRect);
			rectangle(Cropped, CroppedRect.tl(),CroppedRect.br(), Scalar(0, 0, 255), 2);
		}
	}
	if (r_char.size() >= 2)
	{
		for (int i = 0; i < r_char.size() - 1; ++i)
		{
			for (int j = i + 1; j < r_char.size(); ++j)
			{
				Rect temp;
				if (r_char.at(j).x < r_char.at(i).x)
				{
					temp = r_char.at(j);
					r_char.at(j) = r_char.at(i);
					r_char.at(i) = temp;
				}
			}
		}
		//issue should be around here 
		for (int i = 0; i < r_char.size(); i++)
		{
			Mat clone = Cropped(r_char.at(i));
			charc.push_back(clone);
		}
		characters.push_back(charc);
		plates.push_back(Cropped);
		draw_char.push_back(Cropped);
	}

	//imshow("Cropped", Cropped);
	imshow("char", draw_char[0]);


	//imshow("contours", dst);
	//waitKey();
	
	waitKey();

	string savesvm = "svm.txt";
	string imgpath = "D:\\Alphabets";

	bool train = trainSVM(savesvm, imgpath);
	if (train)
	{
		cout << "Training Complete!" << endl;
	}
	else
	{
		cout << "Error during Training...." << endl;
	}
	waitKey();

	for (size_t i = 0; i < characters.size(); i++)
	{
		string result;
		for (size_t j = 0; j < characters.at(i).size(); ++j)
		{
			char plate = character_recognition(characters.at(i).at(j));
			result.push_back(plate);
		}
		text.push_back(result);
		licenseNumber += result;
	}
	cout << "License Plate Number is: " << licenseNumber << endl;
	waitKey();
}

