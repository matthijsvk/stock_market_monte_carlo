// dear imgui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline If you are new to dear imgui, see examples/README.txt
// and documentation at the top of imgui.cpp. (GLFW is a cross-platform general
// purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics
// context creation, etc.)

#define IMPLOT_DISABLE_OBSOLETE_FUNCTIONS

#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "implot.h"
#include "stock_market_monte_carlo/simulations.h"

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load
//  OpenGL function pointers. Helper libraries are often used for this purpose!
//  Here we are supporting a few common ones (gl3w, glew, glad). You may use
//  another loader/header of your choice (glext, glLoadGen, etc.), or chose to
//  manually implement your own.
// Load OpenGL functions
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>  // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>  // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)

#include <glad/glad.h>  // Initialize with gladLoadGL()

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#define GLFW_INCLUDE_NONE       // GLFW including OpenGL headers causes ambiguity or
// multiple definition errors.
#include <glbinding/Binding.h>  // Initialize with glbinding::Binding::initialize()
#include <glbinding/gl/gl.h>

using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#define GLFW_INCLUDE_NONE  // GLFW including OpenGL headers causes ambiguity or
// multiple definition errors.
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>  // Initialize with glbinding::initialize()
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include <atomic>
#include <thread>

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

double rng_uniform(float min, float max) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(min, max);
  return dist(e2);
}

double rng_normal(float mean, float std) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(mean, std);
  return dist(e2);
}

static double cumulative_normal_standard(double d) {
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}

double cumulative_normal(double x, double mu, double s) {
  // simply move and scaleso we can use standard cumulative normal distribution
  return cumulative_normal_standard((x - mu) / s);
}

double cumulative_normal_truncleft(double x, double mu, double s, double a) {
  // https://en.wikipedia.org/wiki/Truncated_normal_distribution
  double alpha = (a - mu) / s;

  // trunc left, not right -> beta = +inf, PHI(beta) = 1.0
  double Z = 1.0 - cumulative_normal_standard(alpha);
  return (cumulative_normal_standard((x - mu) / s) - cumulative_normal_standard(alpha)) / Z;
}

float normal(float x, float m, float s) {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;
  return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

double RationalApproximation(double t) {
  // Abramowitz and Stegun formula 26.2.23.
  // The absolute value of the error should be less than 4.5 e-4.
  double c[] = {2.515517, 0.802853, 0.010328};
  double d[] = {1.432788, 0.189269, 0.001308};
  return t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0);
}

double NormalCDFInverse(double p) {
  // https://www.johndcook.com/blog/cpp_phi_inverse/
  if (p <= 0.0 || p >= 1.0) {
    throw std::invalid_argument(
        fmt::format("Invalid input argument ({:f}); must be larger than 0 but less than 1.", p));
  }

  // See article above for explanation of this section.
  if (p < 0.5) {
    // F^-1(p) = - G^-1(p)
    return -RationalApproximation(sqrt(-2.0 * log(p)));
  } else {
    // F^-1(p) = G^-1(1-p)
    return RationalApproximation(sqrt(-2.0 * log(1 - p)));
  }
}

void update_quartiles(std::vector<float> &quartiles, float &mean, float &std) {
  // todo estimate quartiles from mean/variance, assuming gaussian distribution of final values? (law large numbers)
  std::vector<double> percs = {0.1, 0.25, 0.5, 0.75, 0.9};
  for (int i = 0; i < percs.size(); i++) {
    quartiles[i] = std * float(NormalCDFInverse(percs[i])) + mean;
    // truncuate normal distribution
    quartiles[i] = std::max(float(0.0), std::max(mean - std, quartiles[i]));
  }
}

void update_mean_std(float &mean, float &std, std::vector<float> &means, std::vector<float> &variances) {
  // population mean: just mean of means
  double sum = std::accumulate(means.begin(), means.end(), 0.0);
  mean = sum / means.size();

  // population variance: https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
  // -> average the variances; then you can take square root to get the average standard deviation
  double sum_var = std::accumulate(variances.begin(), variances.end(), 0.0);
  double mean_var = sum / variances.size();
  std = std::sqrt(mean_var);
}

float normal_pdf(float x, float m, float s) {
  static const float inv_sqrt_2pi = 0.3989422804014327;
  float a = (x - m) / s;

  return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

long update_count_below_min(float &min_final_amount, float mean, float std, long n_simulations) {
  // central limit theorem: totals are normally distributed (for large N)
  // -> estimate count from mean/var
  float prob = normal_pdf(min_final_amount, mean, std);
  long count_below_min = n_simulations * prob;
  return count_below_min;
}

int main(int argc, char *argv[]) {
  fmt::print("argc: {}\n", argc);
  long max_n_simulations;
  int n_periods, n_gpus;
  if (argc == 4) {
    char *end;
    n_gpus = atoi(argv[1]);
    n_periods = std::strtol(argv[2], &end, 10);
    max_n_simulations = long(std::strtol(argv[3], &end, 10));
    fmt::print("n_periods: {} | max_n_simulations: {}\n", n_periods, max_n_simulations);
  } else {
    fmt::print(
        "usage: example_gui_simulated <n_gpus> <n_months> <n_simulations>, eg "
        "example_gui_simulated 1 360 100000");
    exit(0);
  }

  //-------------------------------------
  // Monte Carlo stock market simulations
  //-------------------------------------
  float initial_capital = 1000;

  ////simulate by sampling from historical monthly returns
  std::string historical_returns_csv = "data/SP500_monthly_returns.csv";
  std::vector<float> historical_returns = read_historical_returns(historical_returns_csv);
  fmt::print("Number of historical data points from which we can sample: {}\n", historical_returns.size());

  // limit max shown for plotting?
  long max_displayed_plots = 25;

  // buffers to store results
  std::vector<float> means;
  std::vector<float> variances;
  // just for visualization
  // calculate 10x more than we show, so we can do random sample to indicate
  // calculations are still going on
  long max_n_visualisation = 10 * max_displayed_plots;
  std::vector<float> final_values_visualized(max_n_visualisation, -1);
  std::vector<std::vector<float>> mc_data(max_n_visualisation, std::vector<float>(n_periods + 1));

  // GPU MC sampling in background thread:
  // https://hackernoon.com/learn-c-multi-threading-in-5-minutes-8b881c92941f
  std::atomic<long> n_simulations = 0;
  std::thread t1(mc_simulations_gpu_reduceBlock,
                 std::ref(n_simulations),
                 max_n_simulations,
                 n_periods,
                 initial_capital,
                 std::ref(historical_returns),
                 std::ref(means),
                 std::ref(variances),
                 n_gpus);

  long count_below_min;

  //  //DEBUG
  //  t1.join();
  //  count_below_min = 0;
  //  for (long i = 0; i < final_values.size(); i++) {
  ////      fmt::print("final value {}: {}\n", i, final_values[i]);
  //    //    if (final_values[i] == -1)
  //    //      fmt::print("final value[{}] == -1!\n", i);
  //    if (final_values[i] < initial_capital)
  //      count_below_min++;
  //  }
  //  fmt::print("{:d}/{:d} ({:.4f}%) are below target final value",
  //             count_below_min, final_values.size(),
  //             100 * float(count_below_min) / final_values.size());
  //  float mean1,std1;
  //  update_mean_std(mean1, std1, final_values, final_values.size());
  //  fmt::print("mean: {:.2f} | std: {:.2f}", mean1, std1);
  //  exit(0);
  //  //   END DEBUG

  // CPU computes <max_displayed_plots> for visualizlation, saving entire
  // trajectory
  std::atomic<long> n_simulations_visualized = 0;
  std::thread t2(mc_simulations_keepdata,
                 std::ref(n_simulations_visualized),
                 max_n_visualisation,
                 n_periods,
                 initial_capital,
                 std::ref(historical_returns),
                 std::ref(mc_data),
                 std::ref(final_values_visualized));
  t2.join();

  //-------------------------------------
  // GUI stuff
  //-------------------------------------
  glfwSetErrorCallback(glfw_error_callback);

  // Init to set up window
  if (!glfwInit()) return 1;

    // Decide GL+GLSL versions
// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char *glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

  // Create window with graphics context
  int window_width = 1280;
  int window_height = 720;
  GLFWwindow *window =
      glfwCreateWindow(window_width, window_height, "MC Stock Market simulation - GPU", nullptr, nullptr);
  if (window == nullptr) return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
  bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
  bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
  bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
  bool err = false;
  glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
  bool err = false;
  glbinding::initialize([](const char *name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });
#else
  bool err = false;  // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader
                     // is likely to requires some form of initialization.
#endif
  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  // Setup Dear ImGui style
  //  ImGui::StyleColorsDark();
  ImGui::StyleColorsClassic();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Our state
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // Main loop
  // to update count only if something changes
  long prev_n_simulations = -1;
  float prev_min_final_amount = -1;
  // UI user configurable parameters
  float min_final_amount = initial_capital;
  std::vector<float> quartiles(5, 0);
  float mean = -1, std = -1;
  float max_value_slider = 10000;
  while (!glfwWindowShouldClose(window)) {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
    // tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data
    // to your main application.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input
    // data to your main application. Generally you may always pass all
    // inputs to dear imgui, and hide them from your application based on
    // those two flags.
    glfwPollEvents();

    // Start the Dear ImGui frame for the iteration
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // resize ImGui window so it always fits the window size
    int width, height, xpos, ypos;
    glfwGetWindowSize(window, &width, &height);
    glfwGetWindowPos(window, &xpos, &ypos);
    //        fmt::print("width: {}, height: {}\n", width, height);
    ImGui::SetNextWindowSize(ImVec2(width, height));
    ImGui::SetNextWindowPos(ImVec2(0, 0));

    // create ImGui Window
    ImGui::Begin("MC Stock market simulation");

    // horizontal line for desired final amount, and to calculate how many
    // simulations are below this
    ImGui::Indent(0.25 * window_width);
    ImGui::SetNextItemWidth(0.5 * window_width);
    ImGui::SliderFloat("Target?", &min_final_amount, 0, max_value_slider);
    ImGui::SetNextItemWidth(0.5 * window_width);
    ImGui::InputFloat("Slider Value",
                      &min_final_amount,
                      max_value_slider / 100,  // step
                      max_value_slider / 10,   // step_fast
                      "%.1f");
    ImGui::Unindent(0.25 * window_width);

    // recompute only if something changes
    // todo for GPU code, n_simulations isn't ever correct; only when everything's done
    if ((prev_n_simulations != n_simulations) || (prev_min_final_amount != min_final_amount)) {
      fmt::print("Update detected! Recomputing statistics... ");
      if (prev_n_simulations != n_simulations) {
        prev_n_simulations = n_simulations;
        // n_el isn't really accurate...
        update_mean_std(mean, std, means, variances);
        update_quartiles(quartiles, mean, std);
        max_value_slider = 10 * quartiles[3];
        //            *max_element(final_values.begin(), final_values.end());
      }
      if (prev_min_final_amount != min_final_amount) {
        prev_min_final_amount = min_final_amount;
      }
      count_below_min = update_count_below_min(min_final_amount, mean, std, means.size());
      fmt::print("done!\n");
    }

    // Plot
    if (ImPlot::BeginPlot(
            "My Plot",
            ImVec2(-1, height - 200)  // leave some space for text below graph
                                      //                          ,ImPlotFlags_NoLegend | ImPlotAxisFlags_AutoFit)
            )) {
      // Limit number for performance & stop changing if all simulations are done
      if (n_simulations < max_n_simulations && n_simulations_visualized > max_displayed_plots) {
        for (int n_shown = 0; n_shown < max_displayed_plots; n_shown++) {
          if (n_shown >= n_simulations_visualized) break;
          int idx = n_shown;

          // ImPlot::SetupAxes("Time (Months)", "Value");//, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);// |
          // ImPlotAxisFlags_LockMin);
          idx = int(rng_uniform(0, mc_data.size()));
          ImPlot::PlotLine(fmt::format("Simulation {}", idx).c_str(), mc_data.at(idx).data(), mc_data.at(idx).size());
        }
        // plot zero line
        std::vector<float> vec1(n_periods, 0);
        ImPlot::PlotLine("zero", vec1.data(), vec1.size());

        // plot horizontal line of desired final amount
        std::vector<float> vec2(n_periods, min_final_amount);
        ImPlot::PlotLine("MINIMUM", vec2.data(), vec2.size());
      } else {
        ImPlot::PlotHistogram("histogram",
                              means.data(),
                              means.size(),
                              ImPlotBin_Sqrt);  // cumulative, density
      }

      ImPlot::EndPlot();
    }

    ImGui::Text("%s", fmt::format("#simulations: {:d}/{:d}", n_simulations, max_n_simulations).c_str());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);

    // calculate and print `statistics
    ImGui::Text("%s",
                fmt::format("min: {:.2f} | Q1: {:.2f} | median: {:.2f} | Q3: "
                            "{:.2f} | max: {:.2f}",
                            quartiles.at(0),
                            quartiles.at(1),
                            quartiles.at(2),
                            quartiles.at(3),
                            quartiles.at(4))
                    .c_str());

    ImGui::Text("%s", fmt::format("mean: {:.2f} | std: {:.2f}", mean, std).c_str());

    ImGui::Text("%s",
                fmt::format("{:d}/{:d} ({:.4f}%) are below target final value",
                            count_below_min,
                            max_n_simulations,
                            100 * float(count_below_min) / max_n_simulations)
                    .c_str());
    ImGui::End();

    //-------------------------------------------------------------------

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  t1.join();
  t2.join();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
