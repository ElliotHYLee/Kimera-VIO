/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   StereoImuPipeline.cpp
 * @brief  Stereo Visual-Inertial Odometry Pipeline Implementation
 * 
 * OVERVIEW:
 * This file implements the complete stereo VIO pipeline architecture that processes
 * synchronized stereo camera pairs and IMU data to estimate 6DOF pose and 3D structure.
 * The pipeline follows a modular design with separate threads for each processing stage.
 * 
 * PIPELINE ARCHITECTURE:
 * 1. Data Provider Module    → Synchronizes and provides stereo frames + IMU data
 * 2. Frontend Module         → Feature detection, tracking, stereo matching
 * 3. Backend Module          → Factor graph optimization and pose estimation  
 * 4. Mesher Module          → 3D reconstruction and mesh generation (optional)
 * 5. LCD Module             → Loop closure detection (optional)
 * 6. Visualizer Module      → Real-time visualization (optional)
 * 7. Display Module         → GUI rendering and user interaction (optional)
 * 
 * KEY DESIGN PATTERNS:
 * - Producer-Consumer: Modules communicate via thread-safe queues
 * - Observer Pattern: Callback registration for inter-module communication
 * - Factory Pattern: Module creation through specialized factories
 * - RAII: Automatic resource management through unique_ptr
 * 
 * THREAD SAFETY:
 * - Each module runs in its own thread when parallel_run_ is enabled
 * - Thread-safe queues handle data passing between modules
 * - Callbacks ensure proper synchronization and data flow
 * 
 * @author Antoni Rosinol
 * @author Marcus Abate
 */

#include "kimera-vio/pipeline/StereoImuPipeline.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <string>

#include "kimera-vio/backend/VioBackendFactory.h"
#include "kimera-vio/dataprovider/StereoDataProviderModule.h"
#include "kimera-vio/frontend/VisionImuFrontendFactory.h"
#include "kimera-vio/loopclosure/LcdFactory.h"
#include "kimera-vio/mesh/MesherFactory.h"
#include "kimera-vio/utils/Statistics.h"
#include "kimera-vio/utils/Timer.h"
#include "kimera-vio/visualizer/DisplayFactory.h"
#include "kimera-vio/visualizer/Visualizer3D.h"
#include "kimera-vio/visualizer/Visualizer3DFactory.h"

DECLARE_bool(do_coarse_imu_camera_temporal_sync);
DECLARE_bool(do_fine_imu_camera_temporal_sync);

namespace VIO {

StereoImuPipeline::StereoImuPipeline(const VioParams& params,
                                     Visualizer3D::UniquePtr&& visualizer,
                                     DisplayBase::UniquePtr&& displayer,
                                     PreloadedVocab::Ptr&& preloaded_vocab)
    : Pipeline(params), stereo_camera_(nullptr) {
  // ========================================================================
  // STEREO VIO PIPELINE CONSTRUCTION OVERVIEW
  // ========================================================================
  // 
  // HIGH-LEVEL CONSTRUCTION PURPOSE:
  // Initialize a complete stereo VIO pipeline with modular architecture where
  // each processing stage runs independently and communicates via callbacks.
  // The constructor establishes the entire processing chain and thread topology.
  //
  // CONSTRUCTION SEQUENCE:
  // 1. STEREO CAMERA SETUP       → Create stereo camera calibration model
  // 2. DATA PROVIDER MODULE      → Setup synchronized stereo+IMU data streaming  
  // 3. FRONTEND MODULE           → Initialize visual-inertial feature processing
  // 4. BACKEND MODULE            → Create factor graph optimization engine
  // 5. MESHER MODULE (optional)  → 3D reconstruction and mesh generation
  // 6. LCD MODULE (optional)     → Loop closure detection for drift correction
  // 7. VISUALIZER (optional)     → Real-time 3D visualization pipeline
  // 8. DISPLAY MODULE (optional) → GUI rendering and user interaction
  // 9. THREAD LAUNCH             → Start all module threads for parallel execution
  //
  // INTER-MODULE COMMUNICATION:
  // - Callbacks: std::bind creates function objects for data flow
  // - Queues: Thread-safe containers for producer-consumer patterns
  // - Smart Pointers: RAII memory management with shared ownership
  // ========================================================================
  
  // ========================================================================
  // 1. STEREO CAMERA SETUP PHASE
  // ========================================================================
  // Purpose: Create stereo camera model with calibration and rectification
  
  // 1.1 Validate stereo camera configuration
  // Ensures exactly two camera parameter sets are provided (left and right)
  CHECK_EQ(params.camera_params_.size(), 2u)
      << "Need two cameras for StereoImuPipeline.";
      
  // 1.2 Create stereo camera calibration model
  // StereoCamera encapsulates: intrinsics, extrinsics, rectification, stereo baseline
  // shared_ptr enables safe sharing between modules (frontend, backend, mesher)
  stereo_camera_ = std::make_shared<StereoCamera>(params.camera_params_.at(0),  // Left camera
                                                  params.camera_params_.at(1)); // Right camera

  // ========================================================================
  // 2. DATA PROVIDER MODULE SETUP PHASE
  // ========================================================================
  // Purpose: Create synchronized stereo+IMU data streaming with temporal alignment
  
  // 2.1 Create stereo data provider module
  // Handles: stereo frame synchronization, IMU data buffering, temporal alignment
  data_provider_module_ = std::make_unique<StereoDataProviderModule>(
      &frontend_input_queue_,                              // Output queue for synchronized data
      "Stereo Data Provider",                              // Thread name for debugging
      parallel_run_,                                       // Enable/disable threading
      params.frontend_params_.stereo_matching_params_);    // Stereo matching configuration
      
  // 2.2 Configure temporal synchronization strategies
  if (FLAGS_do_coarse_imu_camera_temporal_sync) {
    // Automatic coarse-level time alignment between IMU and camera streams
    // Uses cross-correlation analysis to find optimal time offset
    data_provider_module_->doCoarseImuCameraTemporalSync();
  }
  
  // 2.3 Apply manual time shift corrections
  if (!FLAGS_do_fine_imu_camera_temporal_sync) {
    if (FLAGS_do_coarse_imu_camera_temporal_sync) {
      // Warning: manual shift will be applied ON TOP of automatic alignment
      LOG(WARNING) << "The manually provided IMU time shift will be applied on "
                      "top of whatever the coarse time alignment calculates. "
                      "This may or may not be what you want!";
    }
    // Apply user-specified IMU time shift (from calibration or manual tuning)
    data_provider_module_->setImuTimeShift(imu_params_.imu_time_shift_);
  }

  // 2.4 Configure external odometry time alignment (if available)
  if (params.odom_params_) {
    // Set time shift for external odometry sources (wheel encoders, GPS, etc.)
    data_provider_module_->setExternalOdometryTimeShift(
        params.odom_params_.value().time_shift_s_);
  }

  // 2.5 Register pipeline callback for data processing
  // std::bind creates a function object that calls this->spinOnce() with data
  // This establishes the main data flow from provider to pipeline processing
  data_provider_module_->registerVioPipelineCallback(
      std::bind(&StereoImuPipeline::spinOnce, this, std::placeholders::_1));

  // ========================================================================
  // 3. FRONTEND MODULE SETUP PHASE
  // ========================================================================
  // Purpose: Initialize visual-inertial frontend for feature processing
  
  // 3.1 Create vision-IMU frontend module
  // Handles: feature detection, stereo matching, tracking, IMU preintegration
  vio_frontend_module_ = std::make_unique<VisionImuFrontendModule>(
      &frontend_input_queue_,                              // Input queue from data provider
      parallel_run_,                                       // Threading configuration
      VisionImuFrontendFactory::createFrontend(           // Factory-created frontend instance
          params.frontend_type_,                           // Frontend algorithm type
          params.imu_params_,                              // IMU sensor parameters  
          gtsam::imuBias::ConstantBias(),                  // Initial IMU bias estimate
          params.frontend_params_,                         // Feature detection/tracking params
          stereo_camera_,                                  // Stereo camera calibration
          FLAGS_visualize ? &display_input_queue_ : nullptr, // Optional visualization output
          FLAGS_log_output,                                // Enable detailed logging
          params.odom_params_));                           // External odometry parameters
          
  // 3.2 Setup inter-module callback references
  // Local reference to backend queue for lambda capture (avoids 'this' capture issues)
  auto& backend_input_queue = backend_input_queue_;
  
  // 3.3 Register IMU time shift update callback
  // Allows frontend to dynamically adjust IMU timing based on online calibration
  // Lambda captures data_provider_module_ by reference for time shift updates
  vio_frontend_module_->registerImuTimeShiftUpdateCallback(
      [&](double imu_time_shift_s) {
        data_provider_module_->setImuTimeShift(imu_time_shift_s);
      });
      
  // 3.4 Register frontend output callback for backend communication
  // Lambda function processes frontend output and forwards keyframes to backend
  vio_frontend_module_->registerOutputCallback(
      [&backend_input_queue](const FrontendOutputPacketBase::Ptr& output) {
        // 3.4.1 Safely cast generic frontend output to stereo-specific type
        // std::dynamic_pointer_cast performs runtime type checking
        auto converted_output =
            std::dynamic_pointer_cast<StereoFrontendOutput>(output);
        CHECK(converted_output); // Ensure cast succeeded

        // 3.4.2 Filter for keyframes only (backend optimization is expensive)
        if (converted_output && converted_output->is_keyframe_) {
          // Create backend input packet with all necessary data:
          // - Timestamp for temporal association
          // - Stereo measurements for factor graph constraints  
          // - Preintegrated IMU measurements (PIM)
          // - Raw IMU data for bias estimation
          // - Odometry poses and velocities (if available)
          backend_input_queue.push(std::make_unique<BackendInput>(
              converted_output->stereo_frame_lkf_.timestamp_,        // Keyframe timestamp
              converted_output->status_stereo_measurements_,         // 3D landmark observations
              converted_output->pim_,                                // Preintegrated IMU measurements
              converted_output->imu_acc_gyrs_,                      // Raw IMU acceleration/gyroscope data
              converted_output->body_lkf_OdomPose_body_kf_,         // Relative odometry pose
              converted_output->body_kf_world_OdomVel_body_kf_));   // World-frame odometry velocity
        } else {
          // Non-keyframes are skipped to reduce backend computational load
          VLOG(5) << "Frontend did not output a keyframe, skipping Backend input.";
        }
      });

  // ========================================================================
  // 4. BACKEND MODULE SETUP PHASE
  // ========================================================================
  // Purpose: Create factor graph-based optimization engine for pose estimation
  
  // 4.1 Configure backend output parameters
  // Controls what data the backend should output to downstream modules
  BackendOutputParams backend_output_params(
      static_cast<VisualizationType>(FLAGS_viz_type) !=  // Enable visualization output
          VisualizationType::kNone,
      FLAGS_min_num_obs_for_mesher_points,               // Minimum observations for meshing
      FLAGS_visualize && FLAGS_visualize_lmk_type);      // Landmark visualization settings

  // 4.2 Create VIO backend optimization module
  // Implements factor graph SLAM using GTSAM library for pose and landmark estimation
  CHECK(backend_params_); // Ensure backend parameters are properly initialized
  vio_backend_module_ = std::make_unique<VioBackendModule>(
      &backend_input_queue_,                              // Input queue from frontend
      parallel_run_,                                      // Threading configuration
      BackendFactory::createBackend(                      // Factory-created backend instance
          static_cast<BackendType>(params.backend_type_), // Backend algorithm type (e.g., VIO, stereo VIO)
          stereo_camera_->getBodyPoseLeftCamRect(),       // Camera-body extrinsic calibration
          stereo_camera_->getStereoCalib(),               // Stereo camera calibration parameters
          *backend_params_,                               // Backend-specific optimization parameters
          imu_params_,                                    // IMU sensor and noise parameters
          backend_output_params,                          // Output configuration settings
          FLAGS_log_output,                               // Enable detailed performance logging
          params.odom_params_));                          // External odometry integration parameters
          
  // 4.3 Register backend failure callback
  // Enables graceful handling of backend optimization failures (convergence issues, etc.)
  vio_backend_module_->registerOnFailureCallback(
      std::bind(&StereoImuPipeline::signalBackendFailure, this));
      
  // 4.4 Register IMU bias update callback (backend → frontend communication)  
  // Backend estimates IMU biases and sends updates back to frontend for compensation
  // std::cref creates constant reference wrapper to avoid copying frontend module
  vio_backend_module_->registerImuBiasUpdateCallback(
      std::bind(&VisionImuFrontendModule::updateImuBias,
                std::cref(*CHECK_NOTNULL(vio_frontend_module_.get())), // Constant reference to frontend
                std::placeholders::_1));                               // IMU bias parameter placeholder
                
  // 4.5 Register map update callback (backend → frontend communication)
  // Backend provides updated landmark positions for frontend tracking
  vio_backend_module_->registerMapUpdateCallback(
      std::bind(&VisionImuFrontendModule::updateMap,
                std::cref(*CHECK_NOTNULL(vio_frontend_module_.get())), // Constant reference to frontend  
                std::placeholders::_1));                               // Map update parameter placeholder

  // ========================================================================
  // 5. MESHER MODULE SETUP PHASE (OPTIONAL)
  // ========================================================================
  // Purpose: Create 3D mesh reconstruction from stereo depth and poses
  
  // 5.1 Create mesher module if sparse mesh visualization is requested
  if (static_cast<VisualizationType>(FLAGS_viz_type) ==
      VisualizationType::kMesh2dTo3dSparse) {
    // Initialize 3D mesh reconstruction module
    mesher_module_ = std::make_unique<MesherModule>(
        parallel_run_,                                      // Threading configuration
        MesherFactory::createMesher(                       // Factory-created mesher instance
            MesherType::PROJECTIVE,                        // Projective meshing algorithm
            MesherParams(stereo_camera_->getBodyPoseLeftCamRect(), // Camera extrinsics
                         params.camera_params_.at(0u).image_size_))); // Image dimensions for projection
                         
    // 5.2 Register backend output callback for mesher
    // Backend provides optimized poses and landmarks for mesh reconstruction  
    vio_backend_module_->registerOutputCallback(
        std::bind(&MesherModule::fillBackendQueue,
                  std::ref(*CHECK_NOTNULL(mesher_module_.get())), // Non-const reference to mesher
                  std::placeholders::_1));                      // Backend output parameter

    // 5.3 Register frontend output callback for mesher
    // Frontend provides stereo depth information for mesh generation
    auto& mesher_module = mesher_module_; // Local reference for lambda capture
    vio_frontend_module_->registerOutputCallback(
        [&mesher_module](const FrontendOutputPacketBase::Ptr& output) {
          // Cast and validate stereo frontend output
          auto converted_output =
              std::dynamic_pointer_cast<StereoFrontendOutput>(output);
          CHECK(converted_output); // Ensure successful cast
          
          // Forward stereo depth data to mesher for 3D reconstruction
          CHECK_NOTNULL(mesher_module.get())
              ->fillFrontendQueue(converted_output);
        });
  }

  // ========================================================================
  // 6. LOOP CLOSURE DETECTION MODULE SETUP PHASE (OPTIONAL)
  // ========================================================================
  // Purpose: Detect revisited locations to correct accumulated drift
  
  // 6.1 Create loop closure detection module if enabled
  if (FLAGS_use_lcd) {
    // Initialize place recognition and loop closure detection
    lcd_module_ = std::make_unique<LcdModule>(
        parallel_run_,                                      // Threading configuration
        LcdFactory::createLcd(                             // Factory-created LCD instance
            LoopClosureDetectorType::BoW,                  // Bag-of-Words place recognition
            params.lcd_params_,                            // LCD algorithm parameters
            stereo_camera_->getLeftCamParams(),            // Left camera intrinsics for feature extraction
            stereo_camera_->getBodyPoseLeftCamRect(),      // Camera-body extrinsic transformation
            stereo_camera_,                                // Full stereo camera calibration
            params.frontend_params_.stereo_matching_params_, // Stereo matching for 3D verification
            std::nullopt,                                  // Optional geometric verification parameters
            FLAGS_log_output,                              // Enable detailed LCD logging
            std::move(preloaded_vocab)));                  // Pre-trained vocabulary (moves ownership)
            
    // 6.2 Register backend output callback for LCD
    // Backend provides optimized poses and landmarks for loop closure detection
    vio_backend_module_->registerOutputCallback(
        std::bind(&LcdModule::fillBackendQueue,
                  std::ref(*CHECK_NOTNULL(lcd_module_.get())), // Non-const reference to LCD module
                  std::placeholders::_1));                    // Backend output parameter

    // 6.3 Register frontend output callback for LCD
    // Frontend provides raw features and descriptors for place recognition
    vio_frontend_module_->registerOutputCallback(
        std::bind(&LcdModule::fillFrontendQueue,
                  std::ref(*CHECK_NOTNULL(lcd_module_.get())), // Non-const reference to LCD module
                  std::placeholders::_1));                    // Frontend output parameter
  }

  // ========================================================================
  // 7. VISUALIZER MODULE SETUP PHASE (OPTIONAL)
  // ========================================================================
  // Purpose: Create real-time 3D visualization of VIO results
  
  // 7.1 Create visualizer module if visualization is enabled
  if (FLAGS_visualize) {
    // Initialize 3D visualization and rendering pipeline
    visualizer_module_ = std::make_unique<VisualizerModule>(
        &display_input_queue_,                             // Output queue to display module
        parallel_run_,                                     // Threading configuration
        FLAGS_use_lcd,                                     // Include loop closure visualizations
        // 7.1.1 Use provided visualizer or create new one via factory
        visualizer ? std::move(visualizer)                // Use externally provided visualizer
                   : VisualizerFactory::createVisualizer( // Create default visualizer
                         VisualizerType::OpenCV,           // OpenCV-based visualization backend
                         static_cast<VisualizationType>(FLAGS_viz_type), // Visualization type (2D/3D/mesh)
                         static_cast<BackendType>(params.backend_type_))); // Backend type for display adaptation
                         
    // 7.2 Register backend output callback for visualizer
    // Backend provides optimized trajectory, landmarks, and covariance for display
    vio_backend_module_->registerOutputCallback(
        std::bind(&VisualizerModule::fillBackendQueue,
                  std::ref(*CHECK_NOTNULL(visualizer_module_.get())), // Non-const reference to visualizer
                  std::placeholders::_1));                           // Backend output parameter

    // 7.3 Register frontend output callback for visualizer
    // Frontend provides current frame, features, and tracking status for display
    auto& visualizer_module = visualizer_module_; // Local reference for lambda capture
    vio_frontend_module_->registerOutputCallback(
        [&visualizer_module](const FrontendOutputPacketBase::Ptr& output) {
          // Cast and validate stereo frontend output
          auto converted_output =
              std::dynamic_pointer_cast<StereoFrontendOutput>(output);
          CHECK(converted_output); // Ensure successful cast
          
          // Forward frontend data to visualizer for real-time display
          CHECK_NOTNULL(visualizer_module.get())
              ->fillFrontendQueue(converted_output);
        });

    // 7.4 Register mesher output callback for visualizer (if mesher is enabled)
    if (mesher_module_) {
      // Mesher provides 3D meshes for visualization
      mesher_module_->registerOutputCallback(
          std::bind(&VisualizerModule::fillMesherQueue,
                    std::ref(*CHECK_NOTNULL(visualizer_module_.get())), // Non-const reference to visualizer
                    std::placeholders::_1));                           // Mesher output parameter
    }

    // ======================================================================
    // 8. DISPLAY MODULE SETUP PHASE (OPTIONAL)
    // ======================================================================
    // Purpose: Handle GUI rendering and user interaction in main thread
    
    // 8.1 Create display module for GUI rendering
    // Note: Display runs in main thread to comply with GUI threading requirements
    CHECK(params.display_params_); // Ensure display parameters are provided
    display_module_ = std::make_unique<DisplayModule>(
        &display_input_queue_,                             // Input queue from visualizer
        nullptr,                                           // No additional input queue
        parallel_run_,                                     // Threading configuration (GUI runs in main thread)
        // 8.1.1 Use provided displayer or create new one via factory
        displayer ? std::move(displayer)                  // Use externally provided displayer  
                  : DisplayFactory::makeDisplay(          // Create default displayer
                        params.display_params_->display_type_, // Display backend type (OpenCV, etc.)
                        params.display_params_,            // Display configuration parameters
                        std::bind(&StereoImuPipeline::shutdown, this))); // Shutdown callback for user termination
  }

  // ========================================================================
  // 9. THREAD LAUNCH PHASE
  // ========================================================================
  // Purpose: Start all module threads for parallel execution
  
  // 9.1 Launch all module processing threads
  // If parallel_run_ flag is false, modules will run sequentially instead
  // Each module runs in its own thread with dedicated message queues for communication
  launchThreads();
  
  // ========================================================================
  // CONSTRUCTION COMPLETE - PIPELINE READY FOR OPERATION
  // ========================================================================
  // The stereo VIO pipeline is now fully initialized with:
  // - Synchronized stereo+IMU data streaming
  // - Visual-inertial frontend processing  
  // - Factor graph backend optimization
  // - Optional 3D reconstruction and mesh generation
  // - Optional loop closure detection for drift correction
  // - Optional real-time visualization and GUI interaction
  // - Thread-safe inter-module communication via callbacks and queues
}

}  // namespace VIO
