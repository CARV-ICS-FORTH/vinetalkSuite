/********************************************
 *   Server for multiple caffe instances
 *   
 *
 ********************************************/
#include <string>
#include <map>
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp"
#include "caffe/caffe.hpp"
#include "tonic.h"
#include "vineImgArgs.h"


#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketStream.h"
#include "Poco/Net/SocketAddress.h"
#include "vine_pipe.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vine_pipe.h"

#include "boost/program_options.hpp"
#include "SENNA_utils.h"
#include "SENNA_Hash.h"
#include "SENNA_Tokenizer.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"

#include "tonic.h"

/* fgets max sizes */
#define MAX_TARGET_VB_SIZE 256

vine_pipe_s * vpipe_s;

using namespace ::std;

int main(int argc, char** argv)
{
	vpipe_s = vine_talk_init();	
	int defaultPortNum = 8000;
	int cudaDevice;
	uint64_t resp;

	if (argc != 3)
	{
		cout<<"Add port number and cuda Device (same as in example config)!!"<<endl;
		return 0;
	}
	defaultPortNum = atoi(argv[1]);
	cudaDevice = atoi(argv[2]);

	cudaSetDevice(cudaDevice);
	Poco::Net::ServerSocket srv(defaultPortNum); // does bind + listen


	cout<<__LINE__<<"Controller acceptConnection"<<endl;

	//Start a process for each accelerator thread	
	for (;;)
	{
		Poco::Net::StreamSocket sock = srv.acceptConnection();	
		vine_task_msg_s *vine_task;

		//cout<<__LINE__<<" Controller: vinetask recv"<<endl;
		sock.receiveBytes(&vine_task, sizeof(vine_task));
		//if (!vine_task)
		//	cout<<"Not vinetask!!"<<endl;

		string task_name = ((vine_object_s*)(vine_task->proc))->name;
		/*IMC*/
		if (task_name == "img_classicification")
		{
			//create map to store NN 
			static map<string,Net<float>*> availiabeNNs;
			map<string,Net<float>*>::iterator it;
			Net<float> *imcNet;

			it = availiabeNNs.find("imc");
			if (it == availiabeNNs.end())
			{
				cout<<"NN has not been created yet!!!"<<endl;
				imcNet = new Net<float>("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/configs/imc.prototxt");
				imcNet->CopyTrainedLayersFrom("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/weights/imc.caffemodel");
				availiabeNNs["imc"] = imcNet;
			}
			else
			{
				imcNet = it->second;
			}

			imgArgs *img_args;
			float loss;

			img_args = (imgArgs *)vine_data_deref(vine_task->args.vine_data);


			Caffe::set_phase(Caffe::TEST);
			if (vine_task->type==GPU)
			{
				Caffe::set_mode(Caffe::GPU);
			}
			else if (vine_task->type==CPU)
			{
				Caffe::set_mode(Caffe::CPU);
			}
			else
			{
				cout<<"Not supported yet (if ANY please change it to CPU or GPU)."<<endl;
				LOG(FATAL) << "Unrecognized vinetask type.\n";
			}
			// handle inputs
			reshape(imcNet, img_args->numVine * img_args->sizeVine);

			vector<Blob<float>*> in_blobs = imcNet->input_blobs();

			in_blobs[0]->set_cpu_data((float*)vine_data_deref(vine_task->io[0].vine_data));
			cout<<"SOCKET : IMC execution in GPU"<<endl;
			/* Call the kernel */
			vector<Blob<float>*> out_blobs = imcNet->ForwardPrefilled(&loss);

			//copy results to shm (destination. source)
			memcpy(vine_data_deref(vine_task->io[vine_task->in_count].vine_data), out_blobs[0]->cpu_data(), img_args->numVine * sizeof(float));

			vine_task->state = task_completed;
			vine_data_mark_ready(vpipe_s, vine_task->io[vine_task->in_count].vine_data);

			//cout<<__LINE__<<" Controller: end"<<endl;	
			sock.sendBytes("end", 1);
		}
		/*FACE*/
		else if (task_name == "facial_recognition")
		{
			//create map to store NN 
			static map<string,Net<float>*> availiabeNNs;
			map<string,Net<float>*>::iterator it;
			Net<float> *faceNet;

			it = availiabeNNs.find("face");
			if (it == availiabeNNs.end())
			{
				cout<<"FACE NN has not been created yet!!!"<<endl;
				faceNet = new Net<float>("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/configs/face.prototxt");
				faceNet->CopyTrainedLayersFrom("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/weights/face.caffemodel");
				availiabeNNs["face"] = faceNet;
			}
			else
			{
				faceNet = it->second;
			}

			//std::chrono::time_point<std::chrono::system_clock> start, end;
			imgArgs *img_args;
			float loss;

			img_args = (imgArgs *)vine_data_deref(vine_task->args.vine_data);


			Caffe::set_phase(Caffe::TEST);
			if (vine_task->type==GPU)
			{
				Caffe::set_mode(Caffe::GPU);
			}
			else if (vine_task->type==CPU)
			{
				Caffe::set_mode(Caffe::CPU);
			}
			else
			{
				cout<<"Not supported yet (if ANY please change it to CPU or GPU)."<<endl;
				LOG(FATAL) << "Unrecognized vinetask type.\n";
			}

			//start = std::chrono::system_clock::now();

			// handle inputs
			reshape(faceNet, img_args->numVine * img_args->sizeVine);

			vector<Blob<float>*> in_blobs = faceNet->input_blobs();

			in_blobs[0]->set_cpu_data((float*)vine_data_deref(vine_task->io[0].vine_data));
			cout<<"SOCKET : FACE execution in GPU"<<endl;
			/* Call the kernel */
			vector<Blob<float>*> out_blobs = faceNet->ForwardPrefilled(&loss);

			//copy results to shm (destination. source)
			memcpy(vine_data_deref(vine_task->io[vine_task->in_count].vine_data), out_blobs[0]->cpu_data(), img_args->numVine * sizeof(float));
			vine_task->state = task_completed;
			vine_data_mark_ready(vpipe_s, vine_task->io[vine_task->in_count].vine_data);
		}
		else if (task_name == "digit_recognition")
		{
			//create map to store NN 
			static map<string,Net<float>*> availiabeNNs;
			map<string,Net<float>*>::iterator it;
			Net<float> *digNet;

			it = availiabeNNs.find("dig");
			if (it == availiabeNNs.end())
			{
				cout<<"DIG NN has not been created yet!!!"<<endl;
				digNet = new Net<float>("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/configs/dig.prototxt");
				digNet->CopyTrainedLayersFrom("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/weights/dig.caffemodel");
				availiabeNNs["dig"] = digNet;
			}
			else
			{
				digNet = it->second;
			}

			//std::chrono::time_point<std::chrono::system_clock> start, end;
			imgArgs *img_args;
			float loss;
			img_args = (imgArgs *)vine_data_deref(vine_task->args.vine_data);


			Caffe::set_phase(Caffe::TEST);
			if (vine_task->type==GPU)
			{
				Caffe::set_mode(Caffe::GPU);
			}
			else if (vine_task->type==CPU)
			{
				Caffe::set_mode(Caffe::CPU);
			}
			else
			{
				cout<<"Not supported yet (if ANY please change it to CPU or GPU)."<<endl;
				LOG(FATAL) << "Unrecognized vinetask type.\n";

			}

			//start = std::chrono::system_clock::now();

			// handle inputs
			reshape(digNet, img_args->numVine * img_args->sizeVine);

			vector<Blob<float>*> in_blobs = digNet->input_blobs();

			//bytesSum((float *)vine_data_deref(vine_task->io[0].vine_data), vine_data_size(vine_task->io[0].vine_data));

			in_blobs[0]->set_cpu_data((float*)vine_data_deref(vine_task->io[0].vine_data));
			cout<<"SOCKET : DIG execution in GPU"<<endl;
			/* Call the kernel */
			vector<Blob<float>*> out_blobs = digNet->ForwardPrefilled(&loss);

			//copy results to shm (destination. source)
			memcpy(vine_data_deref(vine_task->io[vine_task->in_count].vine_data), out_blobs[0]->cpu_data(), img_args->numVine * sizeof(float));
			vine_task->state = task_completed;
			vine_data_mark_ready(vpipe_s, vine_task->io[vine_task->in_count].vine_data);

		}
		/*NLP POS*/
		else if (task_name == "pos")
		{
			//create map to store NN 
			static map<string,Net<float>*> availiabeNNs;
			map<string,Net<float>*>::iterator it;
			Net<float> *posNet;

			/* SENNA Inits */
			/* options */
			char *opt_path = NULL;
			int opt_usrtokens = 0;

			/* the real thing */
			char target_vb[MAX_TARGET_VB_SIZE];
			/* inputs */
			SENNA_Hash *word_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/words.lst");
			SENNA_Hash *caps_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/caps.lst");
			SENNA_Hash *suff_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/suffix.lst");
			SENNA_Hash *gazt_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/gazetteer.lst");

			SENNA_Hash *gazl_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.loc.lst","/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.loc.dat");

			SENNA_Hash *gazm_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.msc.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.msc.dat");

			SENNA_Hash *gazo_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.org.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.org.dat");

			SENNA_Hash *gazp_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.per.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.per.dat");

			/* Inputs all move to controller*/
			SENNA_POS *pos = SENNA_POS_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/pos.dat");
			SENNA_CHK *chk = SENNA_CHK_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/chk.dat");
			SENNA_NER *ner = SENNA_NER_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.dat");

			SENNA_Hash *pos_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/pos.lst");

			/* tokenizer */
			SENNA_Tokenizer *tokenizer =
				SENNA_Tokenizer_new(word_hash, caps_hash, suff_hash, gazt_hash, gazl_hash,
						gazm_hash, gazo_hash, gazp_hash, opt_usrtokens);
			static TonicSuiteApp app;

			//string tmp(vine_data_deref(vine_task->io[0].vine_data));
			// tokenize	
			SENNA_Tokens *tokens = SENNA_Tokenizer_tokenize(tokenizer, vine_data_deref(vine_task->io[0].vine_data) );
			app.pl.num = tokens->n;

			if (app.pl.num == 0)
			{
				cout<<"Empty or no tokens found!"<<endl;
				vine_talk_exit();
				exit(-1);
			}

			it = availiabeNNs.find("pos");
			if (it == availiabeNNs.end())
			{
				cout<<"POS NN has not been created yet!!!"<<endl;
				posNet = new Net<float>("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/configs/pos.prototxt");
				posNet->CopyTrainedLayersFrom("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/weights/pos.caffemodel");
				availiabeNNs["pos"] = posNet;
			}
			else
			{
				posNet = it->second;
			}

			app.net = posNet;

			Caffe::set_phase(Caffe::TEST);
			if (vine_task->type==GPU)
			{
				Caffe::set_mode(Caffe::GPU);
				app.gpu = true;
			}
			else if (vine_task->type==CPU)
			{
				Caffe::set_mode(Caffe::CPU);
			}
			else
			{
				cout<<"Not supported yet (if ANY please change it to CPU or GPU)."<<endl;
				LOG(FATAL) << "Unrecognized vinetask type.\n";
			}

			int *pos_output = malloc(app.pl.num * sizeof(int) );


			cout<<"SOCKET : POS execution in GPU"<<endl;
			//inside SENNA_POS_forwadr we do vine_task issue
			pos_output = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, app);

			//Check output: FIX IT!!!
			if (pos_output==-1)
			{
				vine_task->state = task_failed;
				cerr<<"POS failed!!" <<endl;
				cerr << __FILE__ << " Failed at " << __LINE__ << endl;
				return (task_failed);
			}
			//Print the results for verification
			for (int i = 0; i < tokens->n; i++) 
			{
				printf("%15s", tokens->words[i]);
				printf(" ");
				printf("%d", pos_output[i]);
				printf("\t%10s", SENNA_Hash_key(pos_hash, pos_output[i]));
				printf("\n");
			}
			// end of sentence
			printf("\n");
			memcpy(vine_data_deref(vine_task->io[vine_task->in_count].vine_data), pos_output, 
					app.pl.num * sizeof(int));


			cout<<"Vinetask completed!!"<<endl;

			vine_task->state = task_completed;
			vine_data_mark_ready(vpipe_s, vine_task->io[vine_task->in_count].vine_data);

			SENNA_Tokenizer_free(tokenizer);

			SENNA_POS_free(pos);

			SENNA_Hash_free(word_hash);
			SENNA_Hash_free(caps_hash);
			SENNA_Hash_free(suff_hash);
			SENNA_Hash_free(gazt_hash);

			SENNA_Hash_free(gazl_hash);
			SENNA_Hash_free(gazm_hash);
			SENNA_Hash_free(gazo_hash);
			SENNA_Hash_free(gazp_hash);

			sock.sendBytes("end", 1);
		}else if (task_name == "ner")
		{
			//create map to store NN 
			static map<string,Net<float>*> availiabeNNs;
			map<string,Net<float>*>::iterator it;
			Net<float> *nerNet;

			/* SENNA Inits */
			/* options */
			char *opt_path = NULL;
			int opt_usrtokens = 0;

			/* the real thing */
			char target_vb[MAX_TARGET_VB_SIZE];
			/* inputs */
			SENNA_Hash *word_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/words.lst");
			SENNA_Hash *caps_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/caps.lst");
			SENNA_Hash *suff_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/suffix.lst");
			SENNA_Hash *gazt_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/gazetteer.lst");

			SENNA_Hash *gazl_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.loc.lst","/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.loc.dat");

			SENNA_Hash *gazm_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.msc.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.msc.dat");

			SENNA_Hash *gazo_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.org.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.org.dat");

			SENNA_Hash *gazp_hash = SENNA_Hash_new_with_admissible_keys(
					opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.per.lst", "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.per.dat");


			/* Inputs all move to controller*/
			SENNA_NER *ner = SENNA_NER_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/data/ner.dat");


			SENNA_Hash *ner_hash = SENNA_Hash_new(opt_path, "/home1/public/manospavl/vineyardProject/vine-applications/vine_applications/djinn_tonic/djinn-1.0/tonic-suite/nlp/hash/ner.lst");
			/* tokenizer */
			SENNA_Tokenizer *tokenizer =
				SENNA_Tokenizer_new(word_hash, caps_hash, suff_hash, gazt_hash, gazl_hash,
						gazm_hash, gazo_hash, gazp_hash, opt_usrtokens);
			static TonicSuiteApp app;

			// tokenize     
			SENNA_Tokens *tokens = SENNA_Tokenizer_tokenize(tokenizer, vine_data_deref(vine_task->io[0].vine_data) );
			app.pl.num = tokens->n;

			if (app.pl.num == 0)
			{
				cout<<"Empty or no tokens found!"<<endl;
				vine_talk_exit();
				exit(-1);
			}

			it = availiabeNNs.find("ner");
			if (it == availiabeNNs.end())
			{
				cout<<"NER NN has not been created yet!!!"<<endl;
				nerNet = new Net<float>("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/configs/ner.prototxt");
				nerNet->CopyTrainedLayersFrom("../vine-applications/vine_applications/djinn_tonic/djinn-1.0/common/weights/ner.caffemodel");
				availiabeNNs["ner"] = nerNet;
			}
			else
			{
				nerNet = it->second;
			}
			app.net = nerNet;
			float loss;
			Caffe::set_phase(Caffe::TEST);
			if (vine_task->type==GPU)
			{
				Caffe::set_mode(Caffe::GPU);
			}
			else if (vine_task->type==CPU)
			{
				Caffe::set_mode(Caffe::CPU);
			}
			else
			{
				cout<<"Not supported yet (if ANY please change it to CPU or GPU)."<<endl;
				LOG(FATAL) << "Unrecognized vinetask type.\n";
			}

			int *ner_output = malloc(app.pl.num * sizeof(int) );
			int input_size = ner->ll_word_size + ner->ll_caps_size + ner->ll_gazl_size +
				ner->ll_gazm_size + ner->ll_gazo_size + ner->ll_gazp_size;
			app.pl.size = ner->window_size * input_size;

			reshape(app.net, app.pl.num * app.pl.size);

			cout<<"POS execution in GPU."<<endl;
			ner_output = SENNA_NER_forward(ner, tokens->word_idx, tokens->caps_idx,
					tokens->gazl_idx, tokens->gazm_idx,
					tokens->gazo_idx, tokens->gazp_idx, app);
			//Check output: FIX IT!!!
			if (ner_output==-1)
			{       
				vine_task->state = task_failed;
				cerr<<"NER failed!!" <<endl;
				cerr << __FILE__ << " Failed at " << __LINE__ << endl;
				return (task_failed);
			}

			//Print the results for verification
			for (int i = 0; i < tokens->n; i++) 
			{
				printf("%15s", tokens->words[i]); 
				printf(" ");
				printf("%d", ner_output[i]);
				printf("\t%10s", SENNA_Hash_key(ner_hash, ner_output[i]));
				printf("\n");
			}       
			// end of sentence
			printf("\n");
			memcpy(vine_data_deref(vine_task->io[vine_task->in_count].vine_data), ner_output,
					app.pl.num * sizeof(int));


			cout<<"Vinetask completed!!"<<endl;

			vine_task->state = task_completed;
			vine_data_mark_ready(vpipe_s, vine_task->io[vine_task->in_count].vine_data);

			SENNA_Tokenizer_free(tokenizer);

			SENNA_NER_free(ner);

			SENNA_Hash_free(word_hash);
			SENNA_Hash_free(caps_hash);
			SENNA_Hash_free(suff_hash);
			SENNA_Hash_free(gazt_hash);

			SENNA_Hash_free(gazl_hash);
			SENNA_Hash_free(gazm_hash);
			SENNA_Hash_free(gazo_hash);
			SENNA_Hash_free(gazp_hash);
		}	
		else
			cout <<"Not supported in Sockets!!"<<endl;
	}

	return 0;
}
