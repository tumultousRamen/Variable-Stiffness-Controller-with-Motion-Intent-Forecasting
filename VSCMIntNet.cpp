#include <sys/time.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "PositionControlClient.h"
#include "friUdpConnection.h"
#include "friClientApplication.h"
#include <time.h>
#include <chrono>
#include <thread>
#include <sys/shm.h>
#include <eigen3/Eigen/Dense>
#include "UdpServer.h"
#include "TrignoEmgClient.h"
#include <algorithm>
#include <vector>
#include "tcp_client.h"
#include <deque>
#include "mintwrapper.h"

/* Boost filesystem */
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp>

// linear regression - for performance comparison
#include "lsr.h"

// Circle fitting - for performance comparison
#include "circle.h"
#include "circleutils.h"

using json = nlohmann::json;
using namespace torch::indexing;
using namespace std;
using namespace KUKA::FRI;
using namespace Eigen;
using namespace boost::filesystem;

/* IP Address/Port for KONI Connection */
#define DEFAULT_PORTID 30200
#define DEFAULT_IP "192.170.10.2"

#define DEFAULT_TRIALNUMBER 0

/* GUI UDP Server Address/Port */
const std::string 	udp_addr_gui("192.168.0.102");
const int 			udp_port_gui = 50000;

/* Shared Memory Function Prototype */
template<typename T>
T * InitSharedMemory(std::string shmAddr, int nElements);
void CreateOrOpenKukaDataFile(boost::filesystem::ofstream & ofs, path kukaDataFilePath);

const std::string model_name = "traced_mintnet_epochs_10.pt";
const std::string hparam_name = "hyper_params.json";

// Main
int main(int argc, char** argv)
{
    // Circle object
    Circle circle;

    // Initialize MIntWrapper
    MIntWrapper minty(model_name, hparam_name);

    // Chrono time object: keeps track of elapsed time
    //std::chrono:steady_clock::time_point first_StartTime;
    std::chrono:steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::chrono::milliseconds interval(5); // 5 milliseconds time interval

    // TCP Server initiation
    int check_routine_ms = 20; // every 20ms check routine

    TcpClient tcp_client(check_routine_ms);

    //connect to host
    tcp_client.conn("192.168.0.109", 50000);

    // UDP Server address and port hardcoded now -- change later
    UDPServer udp_server(udp_addr_gui, udp_port_gui);

    if(argc < 4)
    {
        printf("Command line arguments: ./VSCMIntNet <Subject Number> <Block Number> <Fitting Method>\n");
        pritnf("<Fitting Method> : 1. Motion Intent Network 2. Circle Fitting 3. Linear Fitting \n");
        exit(1);
    }

    int subjectNumber 	= atoi(argv[1]);
    int blockNum 	= atoi(argv[2]);
    int fitMethod = atoi(argv[3]);

    /* Check Inputted Subject Number */
    if (subjectNumber < 0)
    {
        printf("Subject Number Must Be Non-negative\n");
        printf("Inputted Subject Number: %d\n", subjectNumber);
        exit(1);
    }
    else
    {
        printf("Subject: %d\n", subjectNumber);
    }

    if(fitMethod < 0 || fitMethod > 3)
    {
        printf("Fitting method can be 1 -> Motion Intent Network 2 -> Circle Fitting 3 -> Linear Fitting\n");
        exit(1);
    }

    bool useMintNet = false;
    bool useCircleFit = false;
    bool useLinearFit = false;

    if(fitMethod == 1)
    {
        // Toggle flag
        useMintNet = true;
    }
    else if(fitMethod == 2)
    {
        // Flag to use circle fitting for projected trajectory
        useCircleFit = true;
    }
    else if(fitMethod == 3)
    {
        // Linear Regression
        // add a flag such that projected trajectory is calculated based on linear estimation
        // Simple linear regression model is trained when motion intent is positive for a distance >= rho
        // Unlike Circle fitting object has not been declared before hand in previous implementation
        useLinearFit = true;
    }
    /* Parse the rest of the command line args */
    	int trialNum;
    	int iterNum = 1;
    	int beginiter = 1;
    	int begintrial = 0;
    	std::string emgIpAddr;

    	const double DEFAULT_KP_AP = 25.0;
    	const double DEFAULT_KN_AP = 25.0;
    	const double DEFAULT_KP_ML = 25.0;
    	const double DEFAULT_KN_ML = 25.0;

    	MatrixXd kp(2, 1);
    	MatrixXd kn(2, 1);
    	const double DEFAULT_k_UB = 300;
    	const double DEFAULT_DELTA = 2.94;
    	const double DEFAULT_R = 150;
    	const double DEFAULT_rho_rate = 0.075;
    	double r;
    	double delta;
    	double k_UB;
    	double rho_rate;


    	//read file
    	MatrixXd input(1, 12);

    	MatrixXd b_LB(2, 1); b_LB << 10, 10;//-30, -10;
    	MatrixXd b_UB(2, 1); b_UB << 10, 10;

    	bool useEmg = false;
    	bool kpInputted_ML = false;
    	bool kpInputted_AP = false;
    	bool knInputted_ML = false;
    	bool knInputted_AP = false;
    	bool blbInputted_AP = false;
    	bool bubInputted_AP = false;
    	bool blbInputted_ML = false;
    	bool bubInputted_ML = false;
    	bool trialNumberInputted = false;
    	bool tuneInputted = false;
    	bool deltaInputted = false;
    	bool rInputted = false;
    	bool rhoInputted = false;
    	bool kubInputted = false;

    	// TCP variables
    	bool bStartSignal = false;
    	bool bEndSignal = false;
    	bool Triggerofend = false;

    	// Iteration based algorithm
    	int triggermode = 1;
    	bool newiterflag = true;
    	int numpriorblock = 6;
    	int flagconfig = 1;


	// Check kp and kn values
	kp(1) = DEFAULT_KP_ML;
    kn(1) = DEFAULT_KN_ML;
    kp(0) = DEFAULT_KP_AP;
    kn(0) = DEFAULT_KN_AP;
    delta = DEFAULT_DELTA;
    r = DEFAULT_R;
    k_UB = DEFAULT_k_UB;
    rho_rate = DEFAULT_rho_rate;

    // Check Inputted Trial Number
    if (!trialNumberInputted)
    {
        trialNum = (int) DEFAULT_TRIALNUMBER;
    }

    b_LB(0) = 10.0;
    b_UB(0) = 10.0;
    b_LB(1) = 10.0;
    b_UB(1) = 10.0;

    // Check EMG use
    TrignoEmgClient emgClient;
    if (useEmg)
    {
        emgClient.SetIpAddress(emgIpAddr);
        emgClient.ConnectDataPort();
        emgClient.ConnectCommPort();
        if (emgClient.IsCommPortConnected())
        {
            // Check if sensors are paired
            emgClient.IsSensorPaired(1);
            emgClient.IsSensorPaired(2);
            emgClient.IsSensorPaired(3);
            emgClient.IsSensorPaired(4);
            emgClient.IsSensorPaired(5);
            emgClient.IsSensorPaired(6);

            /* Turns on backwards compatibility and upsampling, then prints the confrimation that it is on, also prints sampling rate*/
            emgClient.CheckBackwards(2);
            emgClient.CheckBackwards(3);
            emgClient.CheckBackwards(4);
            emgClient.CheckBackwards(5);

            emgClient.SendCommand(1); // this command signals the emg server to send readings to Data Port
            std::thread emgReceiveThread(&TrignoEmgClient::ReceiveDataStream, &emgClient);
            emgReceiveThread.detach();
        }
    }

    // Setup damping modes
    MatrixXd groupDampingModes(1, 25);
    MatrixXd pathNumber(1, 25);
    if(!tuneInputted)
    {

        groupDampingModes << 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2;
        pathNumber << 0, 2, 4, 3, 1, 3, 4, 0, 1, 0, 2, 3, 4, 2, 1, 0, 4, 1, 3, 2, 3, 1, 4, 2, 0;
    }
    else
    {
        printf("Tuning\n");
        groupDampingModes << 0, 0, 2, 2, 2, 0, 0, 0, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1;
        pathNumber << 1, 0, 2, 1, 3, 2, 3, 4, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1;
    }

    int pathNum = pathNumber(blockNum-((int) (blockNum/groupDampingModes.cols()))*((int) groupDampingModes.cols()) - 1);
    int groupDamping =  groupDampingModes(blockNum-((int) (blockNum/groupDampingModes.cols()))*((int) groupDampingModes.cols()) - 1);
    printf("Block: %d\n", blockNum);



    // Definition of target points
    MatrixXd targetx(7, 11);
    MatrixXd targety(7, 11);



	if (!tuneInputted)
	{

		targetx << 0, 6, -5, 8, -1, 0, 5.7, -2.2, 3.1, -9, 0,
			   0, 9.6, 0.2, -7, 3.4, 0, 7.3, 2.1, -4, 8.7, 0,
			   0, -9.6, -0.2, 7, -3.4, 0, 9.2, -5.4, 0.4, 5.4, 0,
			   0, -6.4, -1.1, 4.7, -8.9, 0, -9.9, -3.6, -8.6, 5.1, 0,
			   0, -9.6, -0.2, 7, -3.4, 0, 5.7, -2.2, 3.1, -9, 0,
			   0, -6.4, -1.1, 4.7, -8.9, 0, 6, -5, 8, -1, 0,
			   0, -9.9, -3.6, -8.6, 5.1, 0, 9.6, 0.2, -7, 3.4, 0;

		targety << 0, 8, -7, -9, 8, 0, 9.8, 8.5, -10, 5.2, 0,
			   0, 8.4, -3.8, -6.9, 9.7, 0, 5.6, -9.6, 8.7, 9.8, 0,
			   0, 8.2, -7, -6.1, 9.8, 0, 6.8, -5.8, 8.4, -4, 0,
			   0, 5.6, -9.6, 8.7, 9.8, 0, 8.2, -7, -6.1, 9.8, 0,
			   0, 8.2, -7, -6.1, 9.8, 0, 9.8, 8.5, -10, 5.2, 0,
			   0, 5.6, -9.6, 8.7, 9.8, 0, 8, -7, -9, 8, 0,
			   0, 8.2, -7, -6.1, 9.8, 0, 8.4, -3.8, -6.9, 9.7, 0;
	}
	else
	{
		targetx << 0, 6, -5, 8, -1, 0, 5.7, -2.2, 3.1, -9, 0,
			   0, 9.6, 0.2, -7, 3.4, 0, 7.3, 2.1, -4, 8.7, 0,
			   0, -9.6, -0.2, 7, -3.4, 0, 9.2, -5.4, 0.4, 5.4, 0,
			  0, -6.4, -1.1, 4.7, -8.9, 0, -9.9, -3.6, -8.6, 5.1, 0,
			  0, -9.6, -0.2, 7, -3.4, 0, 5.7, -2.2, 3.1, -9, 0,
			  0,-8.7,-1.1,9,-7.9,-0.6,7.2,-5.1,3.9,-8.2,0,
			  0,-8.7,-1.1,9,-7.9,-0.6,7.2,-5.1,3.9,-8.2,0;

		targety << 0, 8, -7, -9, 8, 0, 9.8, 8.5, -10, 5.2, 0,
			   0, 8.4, -3.8, -6.9, 9.7, 0, 5.6, -9.6, 8.7, 9.8, 0,
			   0, 8.2, -7, -6.1, 9.8, 0, 6.8, -5.8, 8.4, -4, 0,
			    0, 5.6, -9.6, 8.7, 9.8, 0, 8.2, -7, -6.1, 9.8, 0,
			    0, 8.2, -7, -6.1, 9.8, 0, 9.8, 8.5, -10, 5.2, 0,
			    0,-5.3,9.5,0.4,8.8,-2.3,3.7,-4.9,5.9,-10,0,
			    0,-5.3,9.5,0.4,8.8,-2.3,3.7,-4.9,5.9,-10,0;
	}

	// K_UB
    MatrixXd KUBprior(6, 2); KUBprior << 700, 900,
                                        500, 500,
                                        700, 300,
                                        500, 300,
                                        700, 900,
                                        300, 900;

    // rho_rate
    MatrixXd rhoRprior(6, 2); rhoRprior << 0.10, 0.05,
                                        0.15, 0.075,
                                        0.15, 0.075,
                                        0.10, 0.05,
                                        0.05, 0.15,
                                        0.10, 0.075;

    int nTrialsPerBlock = targetx.cols();
    if (trialNum >= nTrialsPerBlock || trialNum < 0)
    {
        printf("Trial Number Out of Acceptable Range\n");
        printf("Inputted Trial Number: %d\n", trialNum);
        printf("Min: 0\n");
        printf("Max: %d\n", nTrialsPerBlock -1);
        exit(1);
    }


    // Declaring path and saving file
    path p_base = current_path();
    // Creating base subject directory
    std::string subjectDir = "Subject" + std::to_string(subjectNumber);
    path p_subject = path(p_base.string()) /= path(subjectDir);
    create_directory(p_subject);

    // Creating fitting method specific directories
    std::string fittingDir;

    if(fitMethod == 1)
    {
        fittingDir = "MIntNet";
    }
    else if(fitMethod == 2)
    {
        fittingDir = "CircleFit";
    }
    else
    {
        fittingDir = "Linear";
    }
    path p_subject_fit = path(p_subject.string()) /= path(fittingDir);
    create_directory(p_subject_fit);

    std::string blockDirold = "Block" + std::to_string(blockNum-1);
    std::string blockDir = "Block" + std::to_string(blockNum);
    path p_blockold = path(p_subject_fit.string()) /= path(blockDirold);
    path p_block = path(p_subject_fit.string()) /= path(blockDir);
    create_directory(p_block);

    std::string iterDir;
    path p_iteration;

    std::string trialDir;
    path p_trial;
    path p_kukadata;
    path p_emgdata;
    path p_configdata;
    boost::filesystem::ofstream OutputFile;
    boost::filesystem::ofstream Configfile;

    // EMG File
    std::string emgfilename = "EmgData.txt";

    // Kuka Data File
    std::string kukafilename = "KukaData.txt";

    // Config Data File
    std::string Configfilename = "Config.txt";

    // Reading input
    std::ifstream parameterfile;
    std::string  fileLocation;

    // Measured Torque
    double meas_torque[7];

    // Variables related to target reaching
    MatrixXd endEffectorXY(2, 1); endEffectorXY << 0, 0;
    MatrixXd neutralXY(2, 1); neutralXY << 0, 0;
    MatrixXd targetXY(2, 1); targetXY << 0, 0;
    MatrixXd targetXYold(2, 1); targetXYold << 0, 0;
    MatrixXd temptarget(2, 1); temptarget << 0.1, 0;
    double ycenter = 0.76;
    bool withinErrorBound;
    bool targetReached = false;

    //Variables related to defining new trial
    bool endBlock = false;
    bool startBlock = false;
    bool readyTrials = false;
    bool startTrial = false;
    int trialSettleIterCount = 0;
    //int const trialEndIterCount = 2000;
    int trialEndIterCount;// = 1;
    int trialWaitCounter = 0;
    int trialWaitTime = rand() % 1000 + 500;
    int beep_flag = 0;
    int beep_count = 0;
    int beep = 0;
    int beep_count1 = 0;


    // Command line arguments
    const char* hostname =  DEFAULT_IP;
    int port = DEFAULT_PORTID;

    // Force Related Varibles
    double ftx;						// Force x-direction (filtered)
    double fty;						// Force y-direction (filtered)
    double ftx_un;					// Force x-direction (unfiltered)
    double fty_un;					// Force y-direction (unfiltered)
    double zerox = 0;				// Force x-baseline
    double zeroy = 0;				// Force y-baseline
    double ftx_0 = 0.0;				// Part of force filtering calc
    double fty_0 = 0.0;  			// Part of force filtering calc
    double al = 0.5;				// exponential moving average alpha level

    // Shared Memory Setup
    std::string shmAddr("/home/justin/Desktop/FRI-Client-SDK_Cpp/example/PositionControl2/shmfile2");
    int shmnElements = 2;
    int * data = InitSharedMemory<int>(shmAddr, shmnElements);

    // Force baseline variables/flags
    int firstIt = 0;			// first iteration flag
    int steady = 0;				// Flag to collect average first 2000 samples of forces without moving KUKA

    // Variables related to variable dampings
    float dt = 0.001;
    //MatrixXd b_var(2, 1); b_var << 10,10; //30, 30;					// Variable damping -- Won't be need this!
    MatrixXd b_cons(2, 1); b_cons << 10, 10;                // Should 10 suffice for constant damping?
    MatrixXd Bgroups(2, 1);
    const double DEFAULT_Damping = 10.0; //30.0;                    // Set to this b_cons value as well?

    MatrixXd x_new_filt(6, 1); x_new_filt << 0, 0, 0, 0, 0, 0;
    MatrixXd x_new_filt_old(6, 1); x_new_filt_old << 0, 0, 0, 0, 0, 0;
    MatrixXd xdot_filt(6, 1); xdot_filt << 0, 0, 0, 0, 0, 0;
    MatrixXd xdot_filt_old(6, 1); xdot_filt_old << 0, 0, 0, 0, 0, 0;
    MatrixXd xdotdot_filt(6, 1); xdotdot_filt << 0, 0, 0, 0, 0, 0;
    MatrixXd xdotdot_filt_old(6, 1); xdot_filt_old << 0, 0, 0, 0, 0, 0;

    // Variables related to variable stiffness (and circle fitting)
    double intentsum = 0.0;
    double distAB = 0.0;
    double d1 = 0.0;
    double angleproj = 0.0;
    double rho = rho_rate*0.01*20.0*sqrt(2.0);
    bool flag_startest = false;
    bool flag_var = false;
    bool flag_var2 = false;
    bool firstest = false;
    std::vector<double> y_reserved;
    std::vector<double> x_reserved;

    // Variables related to Motion Intent Network
    std::deque<Eigen::ArrayXXf> input_Mint;
    int const threshold_MIntNet = 5;
    int const input_seq_length = 125;
    bool firstMIntFit = false;

    MatrixXd k_var(2, 1); k_var << 0, 0;
    MatrixXd Kgroups(2, 1);
    MatrixXd xstart(2, 1), xend(2, 1), xcurrent(2, 1), bproj(2, 1), aproj(2, 1), amirror(2, 1), projected(2, 1), projected_lin(2, 1);
    MatrixXd tempstiff(2, 1);
    Eigen::ArrayXf mintnet_current_state(6,1);
    Eigen::ArrayXf mintnet_projections(2, 1);

    //Circle fitting
    std::vector<double> y_test {7.,6.,8.,7.,5.,7.};
    std::vector<double> x_test {1.,2.,5.,7.,9.,3.};
    int ysize;
    double circle_a = 0.0;
    double circle_b = 0.0;
    double circle_r = 0.0;

    //Circle fitting new criteria
    double x_Mean = 0.0;
    double y_Mean = 0.0;
    double dist_to_center = 0.0;
    double ux = 0.0;
    double uy =0.0;
    bool flag_counter = false;
    int counter_estimate = 0;
    int const estimate_threshold = 100;
    int estimate_MIntNet = 0;



    // variables for intentsum
    double displacement = 0.0;
    double displacement_filt = 0.0;
    double displacement_filt_old = 0.0;
    double ddot = 0.0;
    double ddotdot = 0.0;
    double ddot_filt = 0.0;
    double ddot_filt_old = 0.0;
    double ddotdot_filt = 0.0;
    double ddotdot_filt_old = 0.0;


    // Variables related to position, velocity and acceleration filter
    double CUTOFF = 20.0;
    double RC = (CUTOFF * 2 * 3.14);
    double df = 1.0 / dt;
    double alpha_filt = RC / (RC + df);


    // Euler Angles
    double phi_euler = 0;
    double theta_euler = 0;
    double psi_euler = 0;

    // ----------------------Initial DH Parameters------------------------
    MatrixXd alpha(1, 7); alpha << M_PI / 2, -M_PI / 2, -M_PI / 2, M_PI / 2, M_PI / 2, -M_PI / 2, 0;
    MatrixXd a(1, 7); a << 0, 0, 0, 0, 0, 0, 0;
    MatrixXd d(1, 7); d << 0.36, 0, 0.42, 0, 0.4, 0, 0.126;
    MatrixXd theta(1, 7); theta << 0, 0, 0, 0, 0, 0, 0;



    MatrixXd qc(6, 1); qc << 0, 0, 0, 0, 0, 0;					// calculated joint space parameter differences (between current and prev iteration)
    MatrixXd delta_q(6, 1); delta_q << 0, 0, 0, 0, 0, 0;		// (current - initial) joint space parameters
    MatrixXd q_init(6, 1); q_init << 0, 0, 0, 0, 0, 0;			// initial joint space parameter (after steady parameter is met)
    MatrixXd q_freeze(6, 1); q_freeze << 0, 0, 0, 0, 0, 0;		// joint parameters when the trials are ended and it will be freezed here

    MatrixXd x_e(6, 1); x_e << 0, 0, 0, 0, 0, 0;				// end effector equilibrium position
    MatrixXd force(6, 1); force << 0, 0, 0, 0, 0, 0;			// force vector (filtered)
    MatrixXd q_new(6, 1); q_new << 0, 0, 0, 0, 0, 0;			// joint space parameters
    MatrixXd x_new(6, 1); x_new << 0, 0, 0, 0, 0, 0;			// calculated current position and pose
    MatrixXd xdot(6, 1);  xdot << 0, 0, 0, 0, 0, 0;				// first derative of x_new
    MatrixXd xdotdot(6, 1); xdotdot << 0, 0, 0, 0, 0, 0;		// second derativive of x_new

    MatrixXd x_old(6, 1); x_old << 0, 0, 0, 0, 0, 0;			// one iteration old position and pose
    MatrixXd x_oldold(6, 1); x_oldold << 0, 0, 0, 0, 0, 0;		// two iteration old position and pose

    // GUI variables
    double gui_data[40];											// xy coordinates that are send to gui
    memset(gui_data, 0, sizeof(double) * 40);						// xy coordinates that are send to gui

    double rangex_ = -0.18;
    double rangex = 0.18;
    double rangey_ = -0.18;
    double rangey =0.18;
    double d_r;
    if(tuneInputted)
    {
      if(blockNum < 0)//if (blockNum <= 6)
      {
        d_r = 0;
      }
      else
      {
        //d_r = (rangex * 2) / 15;
        d_r = 0.015;
      }
    }
    else
    {
      //d_r = (rangex * 2) / 15;
      d_r = 0.015;
    }
    int unit = 5;
    int trialsec = 0;
    int flag_unitchange = 0;
    //int flag_unitchange2 = 0;
    int unitchange_count = 0;
    //double ex_r = (rangex * 2) / 15;
    double ex_r = 0.015;
    double u_r = 0.005;//ex_r - radius_e;
    int guiMode = 2;
    //int guiMode3 = 1;
    int guiMode3;
    int guiMode2;
    double radius_e = 0.015-0.002;//ex_r - u_r;//0.005;

    if(tuneInputted)
    {
      guiMode2 = 1;//2;
      guiMode3 = 1;//2;
      trialEndIterCount = 1;//2000;
    }
    else
    {
      guiMode2 = 1;
      guiMode3 = 1;
      trialEndIterCount = 1;
    }


    int count = 0;					// iteration counter
    float sampletime = 0.001;
    double MJoint[7] = { 0 };		// measured joint position

    double MaxRadPerSec[7] = { 1.7104,1.7104,1.7453,2.2689,2.4435,3.14159,3.14159 }; //absolute max velocity (no load from KUKA manual for iiwa 800)																					 //double MaxRadPerSec[7]={1.0,1.0,1.0,1.0,1.0,1.0,1.0}; //more conservative velocity limit
    double MaxRadPerStep[7] = { 0 };
    // calculating max step
    for(int i = 0; i < 7; i++)
    {
        MaxRadPerStep[i] = sampletime*MaxRadPerSec[i];
    }

    double MaxJointLimitRad[7] = { 2.9671,2.0944,2.9671,2.0944,2.9671,2.0944,3.0543 };//Max joint limits in radians (can be changed to further restrict motion of robot)
    double MinJointLimitRad[7] = { -2.9671,-2.0944,-2.9671,-2.0944,-2.9671,-2.0944,-3.0543 }; //Min joint limits in radians (can be changed to further restrict motion of robot)

    // create new joint position client
    PositionControlClient client;
    client.InitValues(MaxRadPerStep, MaxJointLimitRad, MinJointLimitRad);

    // create new udp connection for FRI
    UdpConnection connection;

    // pass connection and client to a new FRI client application
    ClientApplication app(connection, client);

    // connect client application to KUKA Sunrise controller
    app.connect(port, hostname);


    // Initialize stiffness, damping, and inertia matrices
    MatrixXd inertia(6, 6);
    MatrixXd stiffness(6, 6);
    MatrixXd damping(6, 6);

    // Initial Joint Angles
    client.NextJoint[0] = -1.5708;
    client.NextJoint[1] = 1.5708;
    client.NextJoint[2] = 0;
    client.NextJoint[3] = 1.5708;
    client.NextJoint[4] = 0;
    client.NextJoint[5] = -1.5708;
    client.NextJoint[6] = -0.958709;
    memcpy(client.LastJoint, client.NextJoint, 7 * sizeof(double));

    while(true)
    {
        // Step through the program
        app.step();

        if(client.KukaState == 4)
        {
            // Count has previously been initialized to zero
            count++;
            // If this is the first iteration
            if(count == 1)
            {
                sampletime = client.GetTimeStep();
                // calculate max step value (as done previously)
                for(int i = 0; i < 7; i++)
                {
                    client.MaxRadPerStep[i] = sampletime*MaxRadPerSec[i];
                }
            }
            // Update measured joint angle values
            memcpy(MJoint, client.GetMeasJoint(), sizeof(double) * 7);

            // Forward Kinematic
            theta << MJoint[0], MJoint[1], MJoint[2], MJoint[3], MJoint[4], MJoint[5], MJoint[6];

            MatrixXd A1(4, 4); A1 << cos(theta(0, 0)), -sin(theta(0, 0))*cos(alpha(0, 0)), sin(theta(0, 0))*sin(alpha(0, 0)), a(0, 0)*cos(theta(0, 0)),
                                     sin(theta(0, 0)), cos(theta(0, 0))*cos(alpha(0, 0)), -cos(theta(0, 0))*sin(alpha(0, 0)), a(0, 0)*sin(theta(0, 0)),
                                     0, sin(alpha(0, 0)), cos(alpha(0, 0)), d(0, 0),
                                     0, 0, 0, 1;
            MatrixXd A2(4, 4); A2 << cos(theta(0, 1)), -sin(theta(0, 1))*cos(alpha(0, 1)), sin(theta(0, 1))*sin(alpha(0, 1)), a(0, 1)*cos(theta(0, 1)),
                                     sin(theta(0, 1)), cos(theta(0, 1))*cos(alpha(0, 1)), -cos(theta(0, 1))*sin(alpha(0, 1)), a(0, 1)*sin(theta(0, 1)),
                                     0, sin(alpha(0, 1)), cos(alpha(0, 1)), d(0, 1),
                                     0, 0, 0, 1;
            MatrixXd A3(4, 4); A3 << cos(theta(0, 2)), -sin(theta(0, 2))*cos(alpha(0, 2)), sin(theta(0, 2))*sin(alpha(0, 2)), a(0, 2)*cos(theta(0, 2)),
                                     sin(theta(0, 2)), cos(theta(0, 2))*cos(alpha(0, 2)), -cos(theta(0, 2))*sin(alpha(0, 2)), a(0, 2)*sin(theta(0, 2)),
                                     0, sin(alpha(0, 2)), cos(alpha(0, 2)), d(0, 2),
                                     0, 0, 0, 1;
            MatrixXd A4(4, 4); A4 << cos(theta(0, 3)), -sin(theta(0, 3))*cos(alpha(0, 3)), sin(theta(0, 3))*sin(alpha(0, 3)), a(0, 3)*cos(theta(0, 3)),
                                     sin(theta(0, 3)), cos(theta(0, 3))*cos(alpha(0, 3)), -cos(theta(0, 3))*sin(alpha(0, 3)), a(0, 3)*sin(theta(0, 3)),
                                     0, sin(alpha(0, 3)), cos(alpha(0, 3)), d(0, 3),
                                     0, 0, 0, 1;
            MatrixXd A5(4, 4); A5 << cos(theta(0, 4)), -sin(theta(0, 4))*cos(alpha(0, 4)), sin(theta(0, 4))*sin(alpha(0, 4)), a(0, 4)*cos(theta(0, 4)),
                                     sin(theta(0, 4)), cos(theta(0, 4))*cos(alpha(0, 4)), -cos(theta(0, 4))*sin(alpha(0, 4)), a(0, 4)*sin(theta(0, 4)),
                                     0, sin(alpha(0, 4)), cos(alpha(0, 4)), d(0, 4),
                                     0, 0, 0, 1;
            MatrixXd A6(4, 4); A6 << cos(theta(0, 5)), -sin(theta(0, 5))*cos(alpha(0, 5)), sin(theta(0, 5))*sin(alpha(0, 5)), a(0, 5)*cos(theta(0, 5)),
                                     sin(theta(0, 5)), cos(theta(0, 5))*cos(alpha(0, 5)), -cos(theta(0, 5))*sin(alpha(0, 5)), a(0, 5)*sin(theta(0, 5)),
                                     0, sin(alpha(0, 5)), cos(alpha(0, 5)), d(0, 5),
                                     0, 0, 0, 1;

            MatrixXd T01(4, 4); T01 << A1;
            MatrixXd T02(4, 4); T02 << T01*A2;
            MatrixXd T03(4, 4); T03 << T02*A3;
            MatrixXd T04(4, 4); T04 << T03*A4;
            MatrixXd T05(4, 4); T05 << T04*A5;
            MatrixXd T06(4, 4); T06 << T05*A6;

            // Inverse Kinematic
            phi_euler = atan2(T06(1, 2), T06(0, 2));
            theta_euler = atan2(sqrt(pow(T06(1, 2), 2) + pow(T06(0, 2), 2)), T06(2, 2));
            psi_euler = atan2(T06(2, 1), -T06(2, 0));

            MatrixXd z0(3, 1); z0 << 0, 0, 1;
            MatrixXd z1(3, 1); z1 << T01(0, 2), T01(1, 2), T01(2, 2);
            MatrixXd z2(3, 1); z2 << T02(0, 2), T02(1, 2), T02(2, 2);
            MatrixXd z3(3, 1); z3 << T03(0, 2), T03(1, 2), T03(2, 2);
            MatrixXd z4(3, 1); z4 << T04(0, 2), T04(1, 2), T04(2, 2);
            MatrixXd z5(3, 1); z5 << T05(0, 2), T05(1, 2), T05(2, 2);
            MatrixXd z6(3, 1); z6 << T06(0, 2), T06(1, 2), T06(2, 2);

            MatrixXd p0(3, 1); p0 << 0, 0, 0;
            MatrixXd p1(3, 1); p1 << T01(0, 3), T01(1, 3), T01(2, 3);
            MatrixXd p2(3, 1); p2 << T02(0, 3), T02(1, 3), T02(2, 3);
            MatrixXd p3(3, 1); p3 << T03(0, 3), T03(1, 3), T03(2, 3);
            MatrixXd p4(3, 1); p4 << T04(0, 3), T04(1, 3), T04(2, 3);
            MatrixXd p5(3, 1); p5 << T05(0, 3), T05(1, 3), T05(2, 3);
            MatrixXd p6(3, 1); p6 << T06(0, 3), T06(1, 3), T06(2, 3);

            MatrixXd J1(6, 1); J1 << z0(1, 0)*(p6(2, 0) - p0(2, 0)) - z0(2, 0)*(p6(1, 0) - p0(1, 0)),
                                    -z0(0, 0)*(p6(2, 0) - p0(2, 0)) + z0(2, 0)*(p6(0, 0) - p0(0, 0)),
                                     z0(0, 0)*(p6(1, 0) - p0(1, 0)) - z0(1, 0)*(p6(0, 0) - p0(0, 0)),
                                     z0(0, 0), z0(1, 0), z0(2, 0);
            MatrixXd J2(6, 1); J2 << z1(1, 0)*(p6(2, 0) - p1(2, 0)) - z1(2, 0)*(p6(1, 0) - p1(1, 0)),
                                    -z1(0, 0)*(p6(2, 0) - p1(2, 0)) + z1(2, 0)*(p6(0, 0) - p1(0, 0)),
                                     z1(0, 0)*(p6(1, 0) - p1(1, 0)) - z1(1, 0)*(p6(0, 0) - p1(0, 0)),
                                     z1(0, 0), z1(1, 0), z1(2, 0);
            MatrixXd J3(6, 1); J3 << z2(1, 0)*(p6(2, 0) - p2(2, 0)) - z2(2, 0)*(p6(1, 0) - p2(1, 0)),
                                    -z2(0, 0)*(p6(2, 0) - p2(2, 0)) + z2(2, 0)*(p6(0, 0) - p2(0, 0)),
                                     z2(0, 0)*(p6(1, 0) - p2(1, 0)) - z2(1, 0)*(p6(0, 0) - p2(0, 0)),
                                     z2(0, 0), z2(1, 0), z2(2, 0);
            MatrixXd J4(6, 1); J4 << z3(1, 0)*(p6(2, 0) - p3(2, 0)) - z3(2, 0)*(p6(1, 0) - p3(1, 0)),
                                    -z3(0, 0)*(p6(2, 0) - p3(2, 0)) + z3(2, 0)*(p6(0, 0) - p3(0, 0)),
                                     z3(0, 0)*(p6(1, 0) - p3(1, 0)) - z3(1, 0)*(p6(0, 0) - p3(0, 0)),
                                     z3(0, 0), z3(1, 0), z3(2, 0);
            MatrixXd J5(6, 1); J5 << z4(1, 0)*(p6(2, 0) - p4(2, 0)) - z4(2, 0)*(p6(1, 0) - p4(1, 0)),
                                    -z4(0, 0)*(p6(2, 0) - p4(2, 0)) + z4(2, 0)*(p6(0, 0) - p4(0, 0)),
                                     z4(0, 0)*(p6(1, 0) - p4(1, 0)) - z4(1, 0)*(p6(0, 0) - p4(0, 0)),
                                     z4(0, 0), z4(1, 0), z4(2, 0);
            MatrixXd J6(6, 1); J6 << z5(1, 0)*(p6(2, 0) - p5(2, 0)) - z5(2, 0)*(p6(1, 0) - p5(1, 0)),
                                    -z5(0, 0)*(p6(2, 0) - p5(2, 0)) + z5(2, 0)*(p6(0, 0) - p5(0, 0)),
                                     z5(0, 0)*(p6(1, 0) - p5(1, 0)) - z5(1, 0)*(p6(0, 0) - p5(0, 0)),
                                     z5(0, 0), z5(1, 0), z5(2, 0);


            // Geometric Jacobian
            MatrixXd Jg(6, 6); Jg << J1, J2, J3, J4, J5, J6;
            MatrixXd Tphi(6, 6); Tphi << 1, 0, 0, 0, 0, 0,
                                         0, 1, 0, 0, 0, 0,
                                         0, 0, 1, 0, 0, 0,
                                         0, 0, 0, 0, -sin(phi_euler), cos(phi_euler)*sin(theta_euler),
                                         0, 0, 0, 0, cos(phi_euler), sin(phi_euler)*sin(theta_euler),
                                         0, 0, 0, 1, 0, cos(theta_euler);

            // Analytical Jacobian
            MatrixXd Ja(6, 6); Ja << Tphi.inverse()*Jg;

            // Set end-effector positioning
            // Frame when subject is positioned at 90 degree rotation from the end-effector
            endEffectorXY(0) = T06(2,3) - ycenter;
            endEffectorXY(1) = -1*T06(0,3);

            // Frame when subject is positioned in-line with the end-effector
            /*
            endEffectorXY(0) = -1*T06(0,3);
            endEffectorXY(1) = -1*T06(2,3) + 2*ycenter;
            */

            steady++;
            if(steady < 2000)
            {
                // for first iteration
                if(firstIt == 0)
                {
                    firstIt = 1;
                    // Initialize equilibrium position and pose
                    x_e << T06(0, 3), T06(1, 3), T06(2, 3), phi_euler, theta_euler, psi_euler;
                    x_old << x_e;
                    x_oldold << x_e;
                    x_new << x_e;

                    // Initializing stiffness, damping and inertia
                    // Should we keep damping values in the plane at 30 Ns/m or 10Ns/m?
                    damping << 10, 0, 0, 0, 0, 0,
                                0, 100, 0, 0, 0, 0,
                                0, 0, 10, 0, 0, 0,
                                0, 0, 0, 0.5, 0, 0,
                                0, 0, 0, 0, 0.5, 0,
                                0, 0, 0, 0, 0, 0.5;

                    // Stiffness and Inertia Init
                    stiffness << 0, 0, 0, 0, 0, 0, //toward varun desk
                                0, 10000000, 0, 0, 0, 0, //up
                                0, 0, 0, 0, 0, 0, //out toward workshop
                                0, 0, 0, 1000000, 0, 0,
                                0, 0, 0, 0, 1000000, 0,
                                0, 0, 0, 0, 0, 1000000;

                    inertia << 10, 0, 0, 0, 0, 0,
                                0, 0.000001, 0, 0, 0, 0,
                                0, 0, 10, 0, 0, 0,
                                0, 0, 0, 0.0001, 0, 0,
                                0, 0, 0, 0, 0.0001, 0,
                                0, 0, 0, 0, 0, 0.0001;
                }
                force << 0, 0, 0, 0, 0, 0;

                zerox = al*(double)data[0] / 1000000 + (1 - al)*zerox;
                zeroy = al*(double)data[1] / 1000000 + (1 - al)*zeroy;

                q_new(0) = -1.5708;
                q_new(1) = 1.5708;
                q_new(2) = 0;
                q_new(3) = 1.5708;
                q_new(4) = 0;
                q_new(5) = -1.5708;

                q_init << q_new;
            }
            // after 2 seconds have elapsed
            else
            {
                // Get force data from shared memory, convert to Newtons
                ftx = (double)data[0] / 1000000 - zerox; //toward varun desk
                ftx_un = (double)data[0] / 1000000 - zerox;

                fty = (double)data[1] / 1000000 - zeroy;
                fty_un = (double)data[1] / 1000000 - zeroy;

                // Filter force data with exponential moving average
                ftx = al*ftx + (1 - al)*ftx_0;
                ftx_0 = ftx;
                fty = al*fty + (1 - al)*fty_0;
                fty_0 = fty;

                force << ftx,0,fty,0,0,0;

                Bgroups << b_cons;
                Kgroups << k_var;

                if(startBlock)
                {
                    targetXY(0) = (targetx(pathNum, trialNum))*0.01;
                    targetXY(1) = (targety(pathNum, trialNum))*0.01;

                    withinErrorBound = (pow((endEffectorXY(0) - targetXY(0)),2) + pow((endEffectorXY(1) - targetXY(1)),2) <= pow(radius_e,2));

                    if (!readyTrials)
                    {
                        if (withinErrorBound)
                        {
                            readyTrials = true;
                        }
                    }
                    else
                    {
                        if (startTrial)
                        {
                            if(beep == 1)
                            {
                              beep_count1++;
                            }
                            if(beep_count1 >= 10)
                            {
                              beep = 0;
                              beep_flag = 1;
                              beep_count1 = 0;
                            }
                            if(beep_flag == 1)
                            {
                              beep_count++;
                            }
                            if(beep_count >=200)
                            {
                              beep_count = 0;
                              beep_flag = 0;
                            }


                            if(flag_unitchange == 1)
                            {
                              unitchange_count++;
                            }
                            if(unitchange_count >=200)
                            {
                              unitchange_count = 0;
                              flag_unitchange = 0;
                            }



                            targetXYold(0) = (targetx(pathNum, trialNum - 1))*0.01;
                            targetXYold(1) = (targety(pathNum, trialNum - 1))*0.01;

                            d_r = ex_r;

                            damping(0, 0) = Bgroups(0);
                            damping(2, 2) = Bgroups(1);
                            stiffness(0, 0) = Kgroups(0);
                            stiffness(2, 2) = Kgroups(1);

                            memcpy(meas_torque, client.GetMeasTorque(), sizeof(double)*7);
                            OutputFile	<< count << " "
                                        << MJoint[0] << " "
                                        << MJoint[1] << " "
                                        << MJoint[2] << " "
                                        << MJoint[3] << " "
                                        << MJoint[4] << " "
                                        << MJoint[5] << " "
                                        << MJoint[6] << " "
                                        << force(0)  << " "
                                        << force(2)  << " "
                                        << x_new(0)  << " "
                                        << x_new(1)  << " "
                                        << x_new(2)  << " "
                                        << x_new(3)  << " "
                                        << x_new(4)  << " "
                                        << x_new(5)  << " "
                                        << damping(0, 0) << " "
                                        << damping(2, 2) << " "
                                        << xdot_filt(0)<< " "
                                        << xdot_filt(2) << " "
                                        << xdotdot_filt(0) << " "
                                        << xdotdot_filt(2) << " "
                                        << targetXY(0) << " "
                                        << targetXY(1) << " "
                                        << targetXYold(0) << " "
                                        << targetXYold(1) << " "
                                        << endEffectorXY(0) << " "
                                        << endEffectorXY(1) << " "
                                        << groupDamping << " "
                                        << meas_torque[0] << " "
                                        << meas_torque[1] << " "
                                        << meas_torque[2] << " "
                                        << meas_torque[3] << " "
                                        << meas_torque[4] << " "
                                        << meas_torque[5] << " "
                                        << meas_torque[6] << " "
                                        << stiffness(0, 0) << " "
                                        << stiffness(2, 2) << " "
                                        << intentsum << " "
                                        << xstart(0) << " "
                                        << xstart(1) << " "
                                        << xend(0) << " "
                                        << xend(1) << " "
                                        << x_e(0) << " "
                                        << x_e(2) << " "
                                        << flag_startest << " "
                                        << flag_var << " "
                                        << flag_var2 << " "
                                        << unit << " "
                                        << flag_unitchange << " "
                                        << targetx(pathNum, unit*trialsec+0)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+0)*0.01 << " "
                                        << targetx(pathNum, unit*trialsec+1)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+1)*0.01 << " "
                                        << targetx(pathNum, unit*trialsec+2)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+2)*0.01 << " "
                                        << targetx(pathNum, unit*trialsec+3)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+3)*0.01 << " "
                                        << targetx(pathNum, unit*trialsec+4)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+4)*0.01 << " "
                                        << targetx(pathNum, unit*trialsec+5)*0.01 << " "
                                        << targety(pathNum, unit*trialsec+5)*0.01 << " "
                                        << circle_a << " "
                                        << circle_b << " "
                                        << circle_r << " "
                                        << b_LB(0) << " "
                                        << b_UB(0) << " "
                                        << kp(0) << " "
                                        << kn(0) << " "
                                        << b_LB(1) << " "
                                        << b_UB(1) << " "
                                        << kp(1) << " "
                                        << kn(1) << " "
                                        << k_UB << " "
                                        << rho << " "
                                        << r << " "
                                        << delta << std::endl;

                            //config file
                            if (flagconfig == 1)
                            {
                              Configfile << p_iteration.string() << std::endl
                                    << folderin2ndpc + "/Iteration" + std::to_string(iterNum) << std::endl
                                    << p_iteration.string() << std::endl
                                    << subjectNumber << std::endl
                                    << blockNum << std::endl
                                    << iterNum << std::endl
                                    << unit << std::endl;
                              flagconfig = 0;
                              if(iterNum==beginiter && trialNum==begintrial + 1)
                              {
                                flagconfig = 1;
                              }

                            }

                            if (!targetReached && withinErrorBound)
                            {
                                targetReached = true;
                                trialSettleIterCount = 0;
                            }

                            // Keep iteration count after target is first reached
                            if (targetReached)
                            {
                                trialSettleIterCount++;

                                // End Trial --- This occurs 2 secs after the target was reached
                                if (trialSettleIterCount >= trialEndIterCount)
                                {
                                    targetReached = false;
                                    startTrial = false;

                                    if(trialNum == unit*iterNum)
                                    {
                                        iterNum++;
                                        bStartSignal = true;
                                        triggermode = 2;
                                        newiterflag = true;
                                    }

                                    // Stop emg recording if necessary
                                    if (useEmg)
                                    {
                                        emgClient.StopWritingFileStream();
                                    }
                                }
                            }


                        }
                        else
                        {
                            // damping can be set here to default value****
                            damping(0, 0) = DEFAULT_Damping;
                            damping(2, 2) = DEFAULT_Damping;

                            if (tuneInputted)
                            {
                              if(blockNum < 0)//if(blockNum <= 6)
                              {
                                targetXY(0) = neutralXY(0);
                                targetXY(1) = neutralXY(1);
                                withinErrorBound = (pow((endEffectorXY(0) - targetXY(0)),2) + pow((endEffectorXY(1) - targetXY(1)),2) <= pow(radius_e,2));
                              }
                            }

                            // Triggered when trial is not running
                            if (!targetReached && withinErrorBound)
                            {
                                targetReached = true;
                                trialWaitCounter = 0;
                                if (trialNum == unit)
                                {
                                    trialWaitTime = rand() % 1000 + 500;
                                }
                                else
                                {
                                  trialWaitTime = 0;
                                }
                                bEndSignal = false;
                                Triggerofend = false;

                            }

                            // Keep iteration count after target is first reached
                            if (targetReached)
                            {
                                trialWaitCounter++;

                                if(blockNum <= numpriorblock)
                                {
                                    triggermode = 1;
                                }

                                if(triggermode == 2)
                                {
                                    Triggerofend = bEndSignal;
                                    if(trialNum == 0)
                                    {
                                        bEndSignal = true;
                                    }

                                }
                                else
                                {
                                    Triggerofend = (trialWaitCounter >= trialWaitTime);
                                }

                                // Start Trial --- This occurs 0.5-1.5 seconds after the neutral position is first reached
                                if (Triggerofend)
                                {
                                    beep = 1;
                                    trialNum++;
                                    if (trialNum < nTrialsPerBlock)
                                    {
                                        targetReached = false;
                                        startTrial = true;
                                        printf("Starting trial %d\n", trialNum);

                                        if (trialNum-1 == unit)
                                        {
                                          trialsec = 1;
                                          flag_unitchange = 1;
                                        }
                                        firstMIntFit = true;
                                        flag_startest = true;
                                        firstest = true;
                                        // I think they should be cleared here too (maybe with firstit)
                                        y_reserved.clear();
                                        x_reserved.clear();
                                        flag_var = false;
                                        flag_var2 = false;
                                        input_Mint.clear();

                                        // Reading parameter file

                                        if(blockNum < numpriorblock || iterNum > 1)
                                        {
                                          fileLocation = p_block.string() + "/Iteration" + std::to_string(iterNum-1) + "/parameters.txt";
                                        }
                                        else
                                        {
                                          fileLocation = p_blockold.string() + "/Iteration" + std::to_string((nTrialsPerBlock-1)/unit) + "/parameters.txt";
                                        }
                                        parameterfile.open(fileLocation);

                                        if (!parameterfile)
                                        {
                                            printf("Unable to open file for Trial: %d\n", iterNum);
                                        }
                                        else
                                        {
                                            parameterfile >> input(0) >> input(1) >> input(2) >> input(3) >> input(4) >> input(5) >> input(6) >> input(7) >> input(8) >> input(9) >> input(10) >> input(11);

                                            if (!kpInputted_AP)
                                            {
                                                kp(0) = input(2);
                                            }
                                            if (!knInputted_AP)
                                            {
                                                kn(0) = input(3);
                                            }
                                            if (!kpInputted_ML)
                                            {
                                                kp(1) = input(6);
                                            }
                                            if (!knInputted_ML)
                                            {
                                                kn(1) = input(7);
                                            }
                                            if (!kubInputted)
                                            {
                                                k_UB = input(8);
                                            }
                                            if (!rhoInputted)
                                            {
                                                rho_rate = input(9);
                                                rho = rho_rate*0.01*20.0*sqrt(2.0);
                                            }
                                            if (!rInputted)
                                            {
                                                r = input(10);
                                            }
                                            if (!deltaInputted)
                                            {
                                                delta = input(11);
                                            }
                                            parameterfile.close();
                                        }
                                        if(blockNum <= numpriorblock)
                                        {
                                          rho_rate = rhoRprior(blockNum-1,iterNum-1);
                                          rho = rho_rate*0.01*20.0*sqrt(2.0);
                                          k_UB = KUBprior(blockNum-1,iterNum-1);
                                        }

                                        // Make iteration directory
                                        iterDir = std::string("Iteration") + std::to_string(iterNum);
                                        p_iteration = path(p_block.string()) /= path(iterDir);
                                        if (newiterflag)
                                        {
                                            create_directory(p_iteration);
                                            newiterflag = false;
                                            flagconfig = 1;
                                            triggermode = 1;
                                            CreateOrOpenKukaDataFile(Configfile, p_configdata);
                                        }

                                        // Make trial directory
                                        trialDir = std::string("Trial") + std::to_string(trialNum);
                                        p_trial = path(p_iteration.string()) /= path(trialDir);
                                        create_directory(p_trial);

                                        if(iterNum == beginiter && trialNum == (begintrial + 2))
                                        {
                                          CreateOrOpenKukaDataFile(Configfile, p_configdata);
                                        }



                                        // Create kuka data trial files
                                        p_kukadata = path(p_trial.string()) /= path(kukafilename);
                                        p_configdata = path(pathconfig) /= path(Configfilename);
                                        CreateOrOpenKukaDataFile(OutputFile, p_kukadata);

                                        // Create emg hdf5 file
                                        if (useEmg)
                                        {
                                            p_emgdata = path(p_trial.string()) /= path(emgfilename);
                                            emgClient.StartWritingFileStream(p_emgdata);
                                        }
                                    }
                                    else
                                    {
                                        q_freeze << q_new;
                                        endBlock = true;
                                        startBlock = false;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    targetXY(0) = temptarget(0);
                    targetXY(1) = temptarget(1);

                    if ((pow((endEffectorXY(0) - targetXY(0)),2) + pow((endEffectorXY(1) - targetXY(1)),2) <= pow(radius_e,2)))
                    {
                        startBlock = true;
                    }
                }
                // Shift old position/pose vectors, calculate new
                x_oldold << x_old;
                x_old << x_new;
                x_new << (inertia/(0.000001) + damping/(0.001) + stiffness).inverse()*(force + (inertia/(0.000001))*(x_old - x_oldold) + stiffness*(x_e - x_old)) + x_old;

                if (x_new(2) >= 0.94)
                {
                  x_new(2) = 0.94;
                }

                if (x_new(2) <= 0.58)
                {
                  x_new(2) = 0.58;
                }

                if (x_new(0) >= 0.18)
                {
                  x_new(0) = 0.18;
                }

                if (x_new(0) <= -0.18)
                {
                  x_new(0) = -0.18;
                }

                qc << Ja.inverse()*(x_new - x_old);
                delta_q << delta_q +qc;
                q_new << delta_q +q_init;

                // Filter
                x_new_filt_old 		= x_new_filt;
                x_new_filt 			= x_new_filt_old + (alpha_filt*(x_new - x_new_filt_old));

                xdot_filt_old 		= xdot_filt;
                xdot 				= (x_new_filt - x_new_filt_old) / dt;
                xdot_filt 			= xdot_filt_old + (alpha_filt*(xdot - xdot_filt_old));

                xdotdot_filt_old 	= xdotdot_filt;
                xdotdot 			= (xdot_filt - xdot_filt_old) / dt;
                xdotdot_filt 		= xdotdot_filt_old + (alpha_filt*(xdotdot - xdotdot_filt_old));

                //user intent sum calculation
                displacement = sqrt(pow(x_new(0),2)+pow(x_new(2),2));

                displacement_filt_old 		= displacement_filt;
                displacement_filt 			= displacement_filt_old + (alpha_filt*(displacement - displacement_filt_old));

                ddot_filt_old 		= ddot_filt;
                ddot 				= (displacement_filt - displacement_filt_old) / dt;
                ddot_filt 			= ddot_filt_old + (alpha_filt*(ddot - ddot_filt_old));

                ddotdot_filt_old 	= ddotdot_filt;
                ddotdot 			= (ddot_filt - ddot_filt_old) / dt;
                ddotdot_filt 		= ddotdot_filt_old + (alpha_filt*(ddotdot - ddotdot_filt_old));
                //Estimating intent of direction & variable stiffness
                  if (startBlock)
                  {
                      intentsum = ddot_filt*ddotdot_filt;
                      if(useMintNet)
                      {
                        mintnet_current_state << x_new(0), x_new(2), xdot(0), xdot(2), xdotdot(0), xdotdot(2);
                        steady_clock::time_point currentTime = steady_clock::now();
                        std::chrono::milliseconds elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
                        Eigen::Array<float, 4, 1> input_Array << xdot(0), xdot(2), xdotdot(0), xdotdot(2);
                        if(elapsedTime < interval)
                        {
                            if(input_Mint.size() < input_seq_length)
                            {
                                input_Mint.push_back(input_Array);
                            }
                            else if(input_Mint.size() >= input_seq_length)
                            {
                                while(input_Mint.size() >= input_seq_length)
                                {
                                    input_Mint.pop_front();
                                }
                                input_Mint.push_back(input_Array);
                            }
                        }
                        else
                        {
                            if(input_Mint.size() < input_seq_length)
                            {
                                input_Mint.push_back(input_Array);
                            }
                            else if(input_Mint.size() > input_seq_length)
                            {
                                while(input_Mint.size() >= input_seq_length)
                                {
                                    input_Mint.pop_front();
                                }
                                input_Mint.push_back(input_Array);
                                // Can call Mint Here after updating the deque with latest values
                                minty.forwardFitSpline(mintnet_current_state, input_Mint);
                                startTime = steady_clock::now;
                            }
                            else
                            {
                                // Call MInt Net here
                                minty.forwardFitSpline(mintnet_current_state, input_Mint);
                                startTime = steady_clock::now();
                            }
                        }
                        mintnet_projections = minty.getEquilibriumPoint();

                        xcurrent << x_new(0), x_new(2);
                        x_e(0) = mintnet_projections(0);
                        x_e(2) = mintnet_projections(1);
                        angleproj = atan2(mintnet_projections(1)-xcurrent(1),mintnet_projections(0)-xcurrent(0));

                        k_var(0) = abs((k_UB/(1 + exp(-r*intentsum + delta)))*cos(angleproj));
                        k_var(1) = abs((k_UB/(1 + exp(-r*intentsum + delta)))*sin(angleproj));
                      }


                      //if ((intentsum >= 0) && (flag_startest)) --> Should we, regardless of desired fitting method, compute the projections from circle and linear fitting?
                      if ((intentsum >= 0) && (flag_startest) && (useCircleFit || useLinearFit))
                      {
                          x_reserved.push_back(x_new(0));
                          y_reserved.push_back(x_new(2));

                          distAB = sqrt(pow(y_reserved[0]-y_reserved[y_reserved.size()-1],2)+pow(x_reserved[0]-x_reserved[x_reserved.size()-1],2));

                          if ((distAB >= rho) && (firstest)) // We could also add the condition on circle and linear flags here
                          {

                              // Linear................
                              simple_linear_regression intdirection(x_reserved, y_reserved, true);
                              intdirection.train();
                              xstart << x_new(0)-0.05, intdirection.predict(x_new(0)-0.05);
                              xend << x_new(0)+0.05, intdirection.predict(x_new(0)+0.05);

                              // Circle.....................
                              ysize = y_reserved.size();
                              double y_reserved_array[ysize];
                              double x_reserved_array[ysize];
                              for (int i=0; i < ysize; i++)
                              {
                                y_reserved_array[i] = y_reserved[i];
                                x_reserved_array[i] = x_reserved[i];
                              }
                              CircleData Datafitcircle(ysize,x_reserved_array, y_reserved_array);
                              circle = CircleFitByPratt(Datafitcircle);
                              circle_a = circle.a;
                              circle_b = circle.b;
                              circle_r = circle.r;
                              cout << "fitdone ("
                                    << circle.a <<","<< circle.b <<")  radius "
                                    << circle.r << endl;

                              flag_var = true;
                              firstest = false;
                              flag_counter = true;

                          }

                          if (flag_counter)
                          {
                            counter_estimate++;

                            if(counter_estimate >= estimate_threshold)
                            {
                                counter_estimate = 0;

                                // Linear................
                                simple_linear_regression intdirection(x_reserved, y_reserved, true);
                                intdirection.train();
                                xstart << x_new(0)-0.05, intdirection.predict(x_new(0)-0.05);
                                xend << x_new(0)+0.05, intdirection.predict(x_new(0)+0.05);


                                // Circle.....................
                                ysize = y_reserved.size();
                                double y_reserved_array[ysize];
                                double x_reserved_array[ysize];
                                for (int i=0; i < ysize; i++)
                                {
                                    y_reserved_array[i] = y_reserved[i];
                                    x_reserved_array[i] = x_reserved[i];
                                }
                                CircleData Datafitcircle(ysize,x_reserved_array, y_reserved_array);
                                circle = CircleFitByPratt(Datafitcircle);
                                circle_a = circle.a;
                                circle_b = circle.b;
                                circle_r = circle.r;
                                cout << "fitdone ("
                                        << circle.a <<","<< circle.b <<")  radius "
                                        << circle.r << endl;
                                flag_var2 = true;
                            }
                            else
                            {
                                flag_var2 = false;
                            }

                          }
                      }
                      if (intentsum < 0)
                      {
                          flag_startest = true;
                          firstest = true;
                          flag_counter = false;
                          counter_estimate = 0;
                          // I think we need them here too
                          y_reserved.clear();
                          x_reserved.clear();
                          flag_var = false;
                          flag_var2 = false;
                      }

                      if (flag_var)
                      {
                          xcurrent << x_new(0), x_new(2);

                          bproj = xend - xstart;
                          aproj = xcurrent - xstart;
                          d1 = (bproj(0)*aproj(0)+bproj(1)*aproj(1))/(pow(bproj(0),2)+pow(bproj(1),2));
                          amirror = bproj*d1;
                          projected_lin = amirror + xstart;

                          // Circle ...........................
                          projected(0) = circle_a + circle_r*(x_new(0)-circle_a)/(sqrt(pow(x_new(0)-circle_a,2)+pow(x_new(2)-circle_b,2)));
                          projected(1) = circle_b + (projected(0)-circle_a)*(x_new(2)-circle_b)/(x_new(0)-circle_a);

                          if(useLinearFit)
                          {
                            x_e(0) = projected_lin(0);
                            x_e(2) = projected_lin(1);
                            angleproj = atan2(projected_lin(1)-xcurrent(1),projected_lin(0)-xcurrent(0));
                          }
                          else if(useCircleFit)
                          {
                            x_e(0) = projected(0);
                            x_e(2) = projected(1);
                            angleproj = atan2(projected(1)-xcurrent(1),projected(0)-xcurrent(0));
                          }

                          k_var(0) = abs((k_UB/(1 + exp(-r*intentsum + delta)))*cos(angleproj));
                          k_var(1) = abs((k_UB/(1 + exp(-r*intentsum + delta)))*sin(angleproj));

                          if(useLinearFit)
                          {
                            k_var(0) = 0.5*k_UB;
                            k_var(1) = 0.5*k_UB;
                          }

                      }
                      else if(!useMintNet)
                      {
                          k_var(0) = 0;
                          k_var(1) = 0;
                      }

                  }


                //b_var(0) = 10.0;
                //b_var(1) = 10.0;

                if (endBlock)
                {
                    q_new << q_freeze;
                }
            }

            // Register new joint angles with KUKA
            client.NextJoint[0] = q_new(0);
            client.NextJoint[1] = q_new(1);
            client.NextJoint[2] = q_new(2);
            client.NextJoint[3] = q_new(3);
            client.NextJoint[4] = q_new(4);
            client.NextJoint[5] = q_new(5);
            client.NextJoint[6] = -0.958709;

            // Send data to pc2 via TCP/IP
            if(bStartSignal)
            {
                tcp_client.ClearStatusBit();
                tcp_client.Start(5000.0, false);
                bStartSignal = false;
            }
            if(tcp_client.WorkDone())
            {
                bEndSignal = true;
                char buffer[40];
                float elapsedTime = tcp_client.GetElapsed_in_sec(false);
                sprintf(buffer, "%.4f", elapsedTime);


            }

            // Send data to visualizer gui
            gui_data[0] = (double) guiMode;
            gui_data[1] = targetXYold(0);
            gui_data[2] = targetXYold(1);
            gui_data[3] = d_r;
            gui_data[4] = endEffectorXY(0);
            gui_data[5] = endEffectorXY(1);
            gui_data[6] = u_r;
            gui_data[7] = targetXY(0);
            gui_data[8] = targetXY(1);
            gui_data[9] = ex_r;
            gui_data[10] = stiffness(2, 2);
            gui_data[11] = (float) (trialNum);
            gui_data[12] = (float) (nTrialsPerBlock - 1);
            gui_data[13] = stiffness(0, 0);
            gui_data[14] = (double) 0;//beep_flag;
            gui_data[15] = (double) guiMode2;

            gui_data[16] = (double) guiMode3;
            gui_data[17] = (double) flag_unitchange;
            gui_data[18] = (double) (unit + 1);
            gui_data[19] = targetx(pathNum, unit*trialsec+0)*0.01;
            gui_data[20] = targety(pathNum, unit*trialsec+0)*0.01;
            gui_data[21] = targetx(pathNum, unit*trialsec+1)*0.01;
            gui_data[22] = targety(pathNum, unit*trialsec+1)*0.01;
            gui_data[23] = targetx(pathNum, unit*trialsec+2)*0.01;
            gui_data[24] = targety(pathNum, unit*trialsec+2)*0.01;
            gui_data[25] = targetx(pathNum, unit*trialsec+3)*0.01;
            gui_data[26] = targety(pathNum, unit*trialsec+3)*0.01;
            gui_data[27] = targetx(pathNum, unit*trialsec+4)*0.01;
            gui_data[28] = targety(pathNum, unit*trialsec+4)*0.01;
            gui_data[29] = targetx(pathNum, unit*trialsec+5)*0.01;
            gui_data[30] = targety(pathNum, unit*trialsec+5)*0.01;

            udp_server.Send(gui_data, 40);
        }
    }

    usleep(10000000);//microseconds //wait for close on other side

    // disconnect from controller
    app.disconnect();
    tcp_client.End(); // TCP/IP

    return 1;
}

/*
Eigen::ArrayXf mintnet(std::string modelPath, std::string hparams, Eigen::ArrayXXf input_a){
    MIntWrapper minty(modelPath, hparams);
    
    std::ifstream f(hparams);
    json data = json::parse(f);
    json data_helper = data["helper_params"];
    json data_model = data["mdl_params"];

    const int input_chn_size = int(data_model["input_size"]);
    const int output_chn_size = int(data_model["output_size"]);
    const int input_seq_length = int(data_helper["input_sequence_length"]);
    const int output_seq_length = int(data_model["M"]) * int(data_model["G"]);

    double max_sim_time = 3.0;
    double sample_rate = 0.005;
    auto sim_timer_start = std::chrono::steady_clock::now();
    auto sample_timer_start = std::chrono::steady_clock::now();

    auto output_minty = minty.forward(input_a);

    Eigen::ArrayXf current_state(6,1);
    current_state << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
    minty.forwardFitSpline(current_state, input_a);
    
    output_eq = minty.getEquilibriumPoint();
    return output_eq;
}
*/

void CreateOrOpenKukaDataFile(boost::filesystem::ofstream & ofs, path kukaDataFilePath)
{
	/* deconstruct kuka file path into path, filename, extension */
	path pp = kukaDataFilePath.parent_path();
	path fname_stem = kukaDataFilePath.stem();
	path fname_ext = kukaDataFilePath.extension();

	/* Make a path to rename old file with same path, and rename if necessary */
	path p_unsuc = path(kukaDataFilePath.string());
	int unsuc_count = 1;
	std::string fname_unsuc;
	if (is_regular_file(p_unsuc))
	{
		while (is_regular_file(p_unsuc)) {
			//fname_unsuc = fname_stem.string() + std::string("_unsuccessful_") + std::to_string(unsuc_count) + fname_ext.string();
			fname_unsuc = fname_stem.string() + std::string("_") + std::to_string(unsuc_count) + fname_ext.string();
			p_unsuc = path(pp.string()) /= path(fname_unsuc);
			unsuc_count++;
		}
		rename(kukaDataFilePath, p_unsuc);
	}

	/* Make file stream */
	ofs.close();
	ofs.open(kukaDataFilePath);
}


// Shared Memory-------------------------------------------------------
template<typename T>
T * InitSharedMemory(std::string shmAddr, int nElements){
	key_t key;
	int shmid;
	size_t shmSize = nElements*sizeof(T);
	T * shm = (T *) malloc(shmSize);
	/* make the key */
	if ((key = ftok(shmAddr.c_str(), 'R')) == -1)
	{
		perror("ftok-->");
		exit(1);
	}

	if ((shmid = shmget(key, shmSize, 0666 | IPC_CREAT)) == -1)
	{
		perror("shmget");
		exit(1);
	}

	shm = (T *) shmat(shmid, (void *)0, 0);

	if (shm == (T *)(-1))
	{
		perror("shmat");
		exit(1);
	}

	for (int i = 0; i<nElements; i++)
	{
		shm[i] = 0.0;
	}

	return shm;
}