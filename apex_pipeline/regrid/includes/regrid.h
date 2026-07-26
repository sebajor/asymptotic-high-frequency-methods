#include <vector>
#include <thread>
#include <array>

constexpr double m_PI = 3.14159265358979323846;
constexpr int init_vector_size {2<<15};

struct Holo_data{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> amp;
    std::vector<double> phase;

    Holo_data(size_t reserve_size){
        x.reserve(reserve_size);
        y.reserve(reserve_size);
        amp.reserve(reserve_size);
        phase.reserve(reserve_size);
    }
};


struct thread_data{
    int x = 0;
    int y = 0;
    double amp = 0;
    double phase = 0;
};

struct hyperparameters {
    std::string in_filename{""};
    std::string out_filename{"output.reg"};
    double resample_map_size{42.f*256}; //map size in arcsec
    double resample_map_size_deg {42.f*256/3600}; //map size in deg
    double gauss_fwhm{26.5f};
    double gauss_fwhm_deg{26.5f/3600};
    float kernel_radius{4.f};
    int max_threads{5};
    int new_samples {256};
};
