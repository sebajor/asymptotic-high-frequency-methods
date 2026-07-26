#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <array>
#include <thread>
#include "../includes/regrid.h"
#include <mutex>
#include <iomanip>

//global mutex
std::mutex mtx;


/* 
 * Amplitude-power relation:
 * The amplitude from the voltmeter is proportional to the sensed voltage at the input.
 * Depending on the attenuation, 1V is mapped to a different RF voltage level.
 * Then the voltage is sampled by the pocket backend that has range  +-10V with 16bits
 * sampling at 10kHz, you can also set a dumptime where the voltmeter will accumulate the 
 * voltage values that it sense in that interval.
 *
 * Then to convert from samples to voltage you have:
 * volt = samples*(20/2**16)/(dumptime*fs)  --> this is the output voltage of the voltmeter
 * rf_power = (volt/att)**2/50              --> calibrated power measurement
 *
 * For the phase the procedure is similar but 1mV is equal to 0.1deg, then
 * phase = volt*1e2;    --> in deg
 *
 */

double pocketBackend2amp(float sample, float dumptime=0.002, float fs=10*1e3){
    double out = sample*20/std::pow(2,16)/(dumptime*fs);
    //out /= att;   //actually not needed
    return out;         
}

float pocketBackend2phase(float sample, float dumptime=0.002, float fs=10*1e3){
    double out = sample*20/std::pow(2,16)/(dumptime*fs); //voltage
    out = out*1e2*m_PI/180;                             //rad
    return out;
}

//
//these are io functions
//
int parse_args(int argc, char* argv[], hyperparameters &params){
    for(int i=1; i<argc; ++i){
        try{
            std::string arg = argv[i];
            if(arg == "--filename" && (i+1) < argc){
                params.in_filename = argv[++i];
            }
            else if(arg == "--map_size" && (i+1)< argc){
                params.resample_map_size = std::stof(argv[++i]);         //in arcsec
                params.resample_map_size_deg = params.resample_map_size/3600;
            }
            else if(arg == "--fwhm"  && (i+1) < argc){
                params.gauss_fwhm = std::stof(argv[++i]);
                params.gauss_fwhm_deg = params.gauss_fwhm/3600;
            }
            else if(arg == "--kernel_rad" && (i+1) < argc){
                params.kernel_radius = std::stof(argv[++i]);     //this is how many fwhm
            }
            else if(arg == "--max_threads" && (i+1)<argc)
                params.max_threads = std::stoi(argv[++i]);
            else if(arg == "--new_samples" && (i+1) < argc)
                params.new_samples = std::stoi(argv[++i]);
        }
        catch(const std::exception &e){
            std::cout << "Exception parse_arg: "<< e.what() <<"\n";
            std::cout << "Error in parameter: "<< argv[i] << "\n";
        }
    }
    return 0;
}

int read_csv(std::string &filename, std::vector<double> &x, std::vector<double> &y, 
        std::vector<double> &amp, std::vector<double> &phase, char delimiter=' ',
        int skip_row=1){
    std::ifstream file(filename.data());
    std::string line;
    std::string cell;
    double tmp {0};
    for(int i=0; i<skip_row; ++i){
        std::getline(file, line);
    }
    try{
        while(std::getline(file, line)){
            std::stringstream ss(line);
            std::getline(ss, cell, delimiter);
            x.push_back(std::stod(cell));
            std::getline(ss, cell, delimiter);
            y.push_back(std::stod(cell));
            std::getline(ss, cell, delimiter);
            tmp = std::stof(cell);
            tmp = pocketBackend2amp(tmp);
            amp.push_back(tmp);
            std::getline(ss, cell, delimiter);
            tmp = std::stod(cell);
            tmp = pocketBackend2phase(tmp);
            phase.push_back(tmp);
        }
    }
    catch(const std::exception& e){
        std::cerr << "Caught std::exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

int write_csv(std::string &filename, Holo_data &output_data){
    std::ofstream file(filename);

    if(!file.is_open()){
        return 1;   //error opening file
    }
    file << std::fixed << std::setprecision(15);    //full precision
    for(size_t i=0; i<output_data.amp.size(); ++i){
        file << output_data.x[i];
        file << " ";
        file << output_data.y[i];
        file << " ";
        file << output_data.amp[i];
        file << " ";
        file << output_data.phase[i];
        file << "\n";
    }
    file.close();
    return 0;
}

//
//  from here on the functions are for the resampling
//


/*  This is the threaded function. 
 *  In this version each thread is in charge of computing the values of a given axis point
 *  ie. you gave a sinlge x_pos and it iterates over all the y_pos
 *  TODO!!!
 */
void resampling_worker_columnwise(
        double x_index,
        double y_pos_i, 
        int y_points,
        double y_step,
        Holo_data &input_data,
        double gaussian_fwhm,
        float kernel_radius,
        Holo_data &output_data
        ){
    //hyperparameters
    double r2_lim = std::pow(gaussian_fwhm*kernel_radius, 2);
    double alpha = 4*std::log(2)/std::pow(gaussian_fwhm,2);              //to have it in std
    //
    double dist2 = 0;        //distance squared of the point to evaluate
    double kernel = 0;
    double amp_sum = 0;
    double cos_sum = 0;
    double sin_sum = 0;
    double kernel_sum =0;    //to normalize
    double x_pos = start+x_index*step;
    double y_pos = y_pos_i;
    for(int j=0; j<y_points; ++j){
        y_pos = y_pos+j*y_step;
        for(size_t i=0; i<input_data.x.size(); ++i){
            dist2 = std::pow(x_pos-input_data.x[i],2)+std::pow(y_pos-input_data.y[i],2);
            if(dist2>r2_lim)
                continue;
            kernel = std::exp(-alpha*dist2);
            amp_sum += kernel*input_data.amp[i];
            cos_sum += kernel*std::cos(input_data.phase[i]);
            sin_sum += kernel*std::sin(input_data.phase[i]);
            kernel_sum += kernel;
        }
        mtx.lock();
        output_data.x[x_index*y_points+j] = x_pos;
        output_data.y[x_index*y_points+j] = y_pos;
        if(kernel_sum !=0){
            output_data.amp[x_index*y_points+j] = amp_sum/kernel_sum;
            output_data.phase[x_index*y_points+j]  = std::atan2(sin_sum, cos_sum);       //here we could do the phase correction right away
        }
        else{

            output_data.amp[x_index*y_points+j] = std::nanf("");
            output_data.phase[x_index*y_points+j]  = std::nanf("");
        }
        mtx.unlock();
    }
}


int resampling_columnwise(double resample_map_size, int N, Holo_data &input_data, 
        Holo_data &output_data,
        double gaussian_fwhm,
        double kernel_radius,
        int max_threads
        ){
    //resize the ouptut data if needed
    output_data.x.resize(N*N);
    output_data.y.resize(N*N);
    output_data.amp.resize(N*N);
    output_data.phase.resize(N*N);
    

    double start = -resample_map_size/2;
    double step = resample_map_size/(N-1);
    std::cout << "map_size:" << resample_map_size << "\n";
    std::cout << "start, step " << start <<","<<step <<"\n";
    double x_resample = 0;
    double y_resample = 0;
    int regrid_x=0, regrid_y=0;

    //containers for the threads and for the outputs
    std::vector<std::thread> thread_pool(max_threads);
    int counter = 0;
    for(int i=0; i<N; ++i){
        if(counter>=max_threads){
            for(int k=0; k<max_threads; ++k){
                thread_pool[k].join();
            }
            counter = 0;
        }
        x_resample = start+i*step;
        thread_pool[counter] = std::thread(
                resampling_worker_columnwise,
                i,
                start,
                N,
                step,
                std::ref(input_data),
                gaussian_fwhm,
                kernel_radius,
                std::ref(output_data)
                );
    }
    return 0;
};


