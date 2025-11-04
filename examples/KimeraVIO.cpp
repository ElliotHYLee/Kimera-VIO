/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   KimeraVIO.cpp
 * @brief  Complete Visual-Inertial Odometry (VIO) Pipeline Example
 * 
 * OVERVIEW:
 * This file demonstrates Kimera-VIO, a real-time metric-semantic SLAM system
 * that combines visual and inertial sensors for robust 6DOF pose estimation.
 * 
 * TECHNICAL ARCHITECTURE:
 * - Frontend: Feature detection, tracking, and visual-inertial initialization
 * - Backend: Factor graph-based optimization using GTSAM library  
 * - Mesher: 3D reconstruction and semantic mesh generation
 * - Visualizer: Real-time 3D trajectory and map visualization
 * 
 * SUPPORTED DATASETS:
 * - EuRoC MAV Dataset: Micro aerial vehicle with stereo+IMU
 * - KITTI Dataset: Automotive stereo vision benchmark
 * 
 * EXECUTION PARADIGMS:
 * - Multi-threaded: Parallel processing using C++11 std::async
 * - Single-threaded: Sequential processing for deterministic debugging
 * 
 * @author Antoni Rosinol
 * @author Luca Carlone
 */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <future>
#include <memory>
#include <utility>

#include "kimera-vio/dataprovider/EurocDataProvider.h"
#include "kimera-vio/dataprovider/KittiDataProvider.h"
#include "kimera-vio/frontend/StereoImuSyncPacket.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/pipeline/MonoImuPipeline.h"
#include "kimera-vio/pipeline/Pipeline.h"
#include "kimera-vio/pipeline/StereoImuPipeline.h"
#include "kimera-vio/utils/Statistics.h"
#include "kimera-vio/utils/Timer.h"

DEFINE_int32(dataset_type,
             0,
             "Type of parser to use:\n "
             "0: Euroc \n 1: Kitti (not supported).");
DEFINE_string(
    params_folder_path,
    "../params/Euroc",
    "Path to the folder containing the yaml files with the VIO parameters.");

int main(int argc, char* argv[]) {
  // ========================================================================
  // KIMERA-VIO PIPELINE EXECUTION OVERVIEW
  // ========================================================================
  // 
  // HIGH-LEVEL TASK PURPOSE:
  // This program demonstrates a complete Visual-Inertial Odometry (VIO) pipeline
  // that estimates camera trajectory and 3D scene structure from:
  // - Visual data: Camera images (mono or stereo)  
  // - Inertial data: IMU measurements (accelerometer + gyroscope)
  //
  // PROCESSING PIPELINE STAGES:
  // 1. INITIALIZATION     → Setup logging, parse configuration parameters
  // 2. DATA PROVIDER      → Create dataset readers (EuRoC, KITTI formats)
  // 3. VIO PIPELINE       → Create mono/stereo processing pipelines  
  // 4. CALLBACK SETUP     → Link data streams between provider and pipeline
  // 5. EXECUTION          → Run pipeline (parallel/sequential modes)
  // 6. ANALYSIS & CLEANUP → Performance metrics and graceful shutdown
  //
  // EXECUTION MODES:
  // - Parallel Mode: Multi-threaded execution using std::async for performance
  // - Sequential Mode: Single-threaded execution for debugging/testing
  //
  // KEY TECHNOLOGIES:
  // - std::async: Asynchronous task execution for parallel processing
  // - std::bind: Function object creation for callback registration  
  // - std::dynamic_pointer_cast: Safe runtime type conversion
  // - GTSAM: Factor graph-based backend optimization
  // - OpenCV: Computer vision algorithms and data structures
  // ========================================================================
  
  // ========================================================================
  // 1. INITIALIZATION PHASE
  // ========================================================================
  // Purpose: Set up the runtime environment, logging, and configuration
  
  // 1.1 Initialize Google's command-line flag parsing library
  // This processes command-line arguments defined with DEFINE_* macros above
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  // 1.2 Initialize Google's logging library (glog)
  // Sets up structured logging with severity levels (INFO, WARNING, ERROR, FATAL)
  google::InitGoogleLogging(argv[0]);

  // 1.3 Parse VIO (Visual-Inertial Odometry) parameters from configuration files
  // Loads YAML configuration files from the specified folder containing:
  // - Camera calibration parameters, IMU parameters, frontend/backend settings
  VIO::VioParams vio_params(FLAGS_params_folder_path);

  // ========================================================================
  // 2. DATA PROVIDER SETUP PHASE
  // ========================================================================
  // Purpose: Create appropriate data provider based on dataset type and sensor configuration
  
  // 2.1 Initialize dataset parser pointer
  // DataProviderInterface abstracts different dataset formats (EuRoC, KITTI, etc.)
  VIO::DataProviderInterface::Ptr dataset_parser = nullptr;
  
  // 2.2 Select dataset parser based on command-line dataset type flag
  switch (FLAGS_dataset_type) {
    case 0: { // EuRoC dataset format
      // 2.2.1 Further specialize based on sensor configuration (mono vs stereo)
      switch (vio_params.frontend_type_) {
        case VIO::FrontendType::kMonoImu: {
          // Monocular camera + IMU configuration
          // MonoEurocDataProvider handles single camera stream with IMU data
          dataset_parser =
              std::make_unique<VIO::MonoEurocDataProvider>(vio_params);
        } break;
        case VIO::FrontendType::kStereoImu: {
          // Stereo camera + IMU configuration
          // EurocDataProvider handles left/right camera streams with IMU data
          dataset_parser = std::make_unique<VIO::EurocDataProvider>(vio_params);
        } break;
        default: {
          LOG(FATAL) << "Unrecognized Frontend type: "
                     << VIO::to_underlying(vio_params.frontend_type_)
                     << ". 0: Mono, 1: Stereo.";
        }
      }
    } break;
    case 1: { // KITTI dataset format
      // KittiDataProvider handles KITTI dataset's specific format and structure
      dataset_parser = std::make_unique<VIO::KittiDataProvider>();
    } break;
    default: {
      LOG(FATAL) << "Unrecognized dataset type: " << FLAGS_dataset_type << "."
                 << " 0: EuRoC, 1: Kitti.";
    }
  }
  
  // 2.3 Ensure data provider was successfully created
  CHECK(dataset_parser);

  // ========================================================================
  // 3. VIO PIPELINE SETUP PHASE
  // ========================================================================
  // Purpose: Create and configure the Visual-Inertial Odometry processing pipeline
  
  // 3.1 Initialize VIO pipeline pointer
  // Pipeline manages the entire VIO processing chain: frontend → backend → output
  VIO::Pipeline::Ptr vio_pipeline;

  // 3.2 Create pipeline instance based on sensor configuration
  switch (vio_params.frontend_type_) {
    case VIO::FrontendType::kMonoImu: {
      // Monocular VIO pipeline - processes single camera + IMU
      // Handles feature detection, tracking, and pose estimation from one camera
      vio_pipeline = std::make_unique<VIO::MonoImuPipeline>(vio_params);
    } break;
    case VIO::FrontendType::kStereoImu: {
      // Stereo VIO pipeline - processes stereo camera pair + IMU  
      // Enables depth estimation from stereo matching and more robust tracking
      vio_pipeline = std::make_unique<VIO::StereoImuPipeline>(vio_params);
    } break;
    default: {
      LOG(FATAL) << "Unrecognized Frontend type: "
                 << VIO::to_underlying(vio_params.frontend_type_)
                 << ". 0: Mono, 1: Stereo.";
    } break;
  }

  // ========================================================================
  // 4. CALLBACK REGISTRATION PHASE  
  // ========================================================================
  // Purpose: Establish communication channels between data provider and VIO pipeline
  
  // 4.1 Register shutdown callback for graceful termination
  // When VIO pipeline shuts down, it will automatically shutdown the data provider
  // std::bind creates a function object that calls dataset_parser->shutdown()
  vio_pipeline->registerShutdownCallback(
      std::bind(&VIO::DataProviderInterface::shutdown, dataset_parser));

  // 4.2 Register IMU data callback
  // Links IMU data stream from data provider to VIO pipeline's IMU queue
  // std::placeholders::_1 allows the callback to accept the IMU data parameter
  dataset_parser->registerImuSingleCallback(std::bind(
      &VIO::Pipeline::fillSingleImuQueue, vio_pipeline, std::placeholders::_1));
  
  // 4.3 Register left camera frame callback
  // Links left camera frames to VIO pipeline's frame processing queue
  // Using blocking variants prevents queue overflow in offline processing
  // (use non-blocking versions for real-time sensor streams)
  dataset_parser->registerLeftFrameCallback(std::bind(
      &VIO::Pipeline::fillLeftFrameQueue, vio_pipeline, std::placeholders::_1));

  // 4.4 Register right camera frame callback (stereo-specific)
  if (vio_params.frontend_type_ == VIO::FrontendType::kStereoImu) {
    // Cast generic pipeline to stereo-specific pipeline for right frame access
    // std::dynamic_pointer_cast safely converts base class pointer to derived class
    auto stereo_pipeline =
        std::dynamic_pointer_cast<VIO::StereoImuPipeline>(vio_pipeline);
    CHECK(stereo_pipeline); // Ensure cast succeeded
    
    // Register right camera frame callback for stereo depth computation
    // Right frames are synchronized with left frames for stereo matching
    dataset_parser->registerRightFrameCallback(
        std::bind(&VIO::StereoImuPipeline::fillRightFrameQueue,
                  stereo_pipeline,
                  std::placeholders::_1));
  }

  // ========================================================================
  // 5. PIPELINE EXECUTION PHASE
  // ========================================================================
  // Purpose: Run the VIO processing pipeline with chosen execution mode
  
  // 5.1 Start performance timing
  auto tic = VIO::utils::Timer::tic();
  bool is_pipeline_successful = false;
  
  // 5.2 Choose execution mode: parallel vs sequential
  if (vio_params.parallel_run_) {
    // ====================================================================
    // 5.2.A PARALLEL EXECUTION MODE
    // ====================================================================
    // Multi-threaded execution for better performance on multi-core systems
    
    // 5.2.A.1 Launch data provider in separate thread using std::async
    // std::launch::async forces asynchronous execution (not deferred)
    // Returns std::future<bool> that can be waited on for completion
    auto handle = std::async(
        std::launch::async, &VIO::DataProviderInterface::spin, dataset_parser);
        
    // 5.2.A.2 Launch VIO pipeline processing in separate thread
    // Handles feature tracking, pose estimation, and backend optimization
    auto handle_pipeline =
        std::async(std::launch::async, &VIO::Pipeline::spin, vio_pipeline);
        
    // 5.2.A.3 Launch shutdown monitor in separate thread
    // Monitors for completion condition and handles graceful shutdown
    // Lambda function checks if data provider has no more data
    auto handle_shutdown = std::async(
        std::launch::async,
        &VIO::Pipeline::waitForShutdown,
        vio_pipeline,
        [&dataset_parser]() -> bool { return !dataset_parser->hasData(); }, // Completion condition
        500,  // Check interval in milliseconds
        true); // Enable force shutdown
        
    // 5.2.A.4 Run visualization in main thread (blocking call)
    // Displays real-time VIO results, trajectory, and point clouds
    vio_pipeline->spinViz();
    
    // 5.2.A.5 Wait for all threads to complete and collect results
    // get() blocks until the async task completes and returns result
    is_pipeline_successful = !handle.get();        // Data provider success
    handle_shutdown.get();                         // Shutdown completion
    handle_pipeline.get();                         // Pipeline completion
    
  } else {
    // ====================================================================
    // 5.2.B SEQUENTIAL EXECUTION MODE  
    // ====================================================================
    // Single-threaded execution for debugging or resource-constrained systems
    
    // 5.2.B.1 Run data provider and pipeline in lockstep
    // Both spin() methods return false when processing should stop
    while (dataset_parser->spin() && vio_pipeline->spin()) {
      continue; // Keep processing until either component signals completion
    };
    
    // 5.2.B.2 Explicit shutdown for sequential mode
    vio_pipeline->shutdown();
    is_pipeline_successful = true;
  }

  // ========================================================================
  // 6. PERFORMANCE ANALYSIS AND CLEANUP PHASE
  // ========================================================================
  // Purpose: Measure performance, log results, and perform graceful shutdown
  
  // 6.1 Calculate total execution time
  // Timer::toc() returns duration since tic() in milliseconds
  auto spin_duration = VIO::utils::Timer::toc(tic);
  
  // 6.2 Output execution statistics
  // LOG(WARNING) ensures timing info is always visible (higher priority than INFO)
  LOG(WARNING) << "Spin took: " << spin_duration.count() << " ms.";
  LOG(INFO) << "Pipeline successful? "
            << (is_pipeline_successful ? "Yes!" : "No!");

  // 6.3 Log detailed performance metrics (success case only)
  if (is_pipeline_successful) {
    // PipelineLogger writes timing data to CSV files for analysis
    // Includes frame processing times, backend optimization times, etc.
    VIO::PipelineLogger logger;
    logger.logPipelineOverallTiming(spin_duration);
  }

  // 6.4 Return appropriate exit code
  // EXIT_SUCCESS (0) indicates successful execution
  // EXIT_FAILURE (1) indicates pipeline failure or error
  return is_pipeline_successful ? EXIT_SUCCESS : EXIT_FAILURE;
}
