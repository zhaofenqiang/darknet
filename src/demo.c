#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network *net;
static image raw_image;
static image raw_image_letter;
//static image buff [3];
//static image buff_letter[3];
//static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
//static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
//static int running = 0;
static double running_time = 0;
//static int demo_frame = 3;
static int demo_detections = 0;
//static float **predictions;
//static int demo_index = 0;
static int demo_done = 0;
static int count = 0;
//static float *avg;
double demo_time;

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh; //if none, default 0;
    demo_hier = hier;  //if none, default 0.5
	float nms = .4;
    printf("Handpose Detection Start\n");

    srand(2222222);

    cap = cvCaptureFromCAM(cam_index);
    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
    	cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
	}
    //cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, 3);
    if(!cap) error("Couldn't connect to webcam.\n");
    if(!prefix){
        cvNamedWindow("Handpose", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Handpose", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Handpose", 0, 0);
            cvResizeWindow("Handpose", 640, 480);
        }
    }

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    raw_image = get_image_from_stream(cap);
    raw_image_letter = letterbox_image(raw_image, net->w, net->h);
    ipl = cvCreateImage(cvSize(raw_image.w, raw_image.h), IPL_DEPTH_8U, raw_image.c);

    while(!demo_done){
        int status = fill_image_from_stream(cap, raw_image);        //从摄像头读取数据
        letterbox_image_into(raw_image, net->w, net->h, raw_image_letter); //将原图像放进灰色框里
        if(status == 0) demo_done = 1;

		if (count == 0) {
			float *X = raw_image_letter.data;

			//算法开始
			demo_time = what_time_is_it_now();
			float *prediction = network_predict(net, X);
			if (l.type == DETECTION) {
				get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
			} else if (l.type == REGION) {
				get_region_boxes(l, raw_image.w, raw_image.h, net->w, net->h,
						demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
			} else {
				error("Last layer must produce detections\n");
			}
			if (nms > 0)
				do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);

			//算法结束
			running_time = what_time_is_it_now() - demo_time;
			printf("\nTime consuming: %.3f seconds.\n", running_time);  //打印所花时间
		}
		count = (count + 1) % 2;

		draw_detections(raw_image, demo_detections, demo_thresh, boxes, probs,
				0, demo_names, demo_alphabet, demo_classes);
		show_image_cv(raw_image, "Handpose", ipl);
		int c = cvWaitKey(1);
		if (c == 27) {
			demo_done = 1;
		}
	}
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
