// dear imgui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline If you are new to dear imgui, see examples/README.txt
// and documentation at the top of imgui.cpp. (GLFW is a cross-platform general
// purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics
// context creation, etc.)

#define IMPLOT_DISABLE_OBSOLETE_FUNCTIONS

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot.h"
//#include "implot_demo.cpp"
#include <fmt/core.h>

#include <simulations.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <chrono>

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load
//  OpenGL function pointers. Helper libraries are often used for this purpose!
//  Here we are supporting a few common ones (gl3w, glew, glad). You may use
//  another loader/header of your choice (glext, glLoadGen, etc.), or chose to
//  manually implement your own.
// Load OpenGL functions
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h> // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h> // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)

#include <glad/glad.h> // Initialize with gladLoadGL()

#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#define GLFW_INCLUDE_NONE // GLFW including OpenGL headers causes ambiguity or
// multiple definition errors.
#include <glbinding/Binding.h> // Initialize with glbinding::Binding::initialize()
#include <glbinding/gl/gl.h>

using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#define GLFW_INCLUDE_NONE // GLFW including OpenGL headers causes ambiguity or
// multiple definition errors.
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h> // Initialize with glbinding::initialize()
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>
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

std::vector<float> quartile_sort(std::vector<float> &vec) {
  // for find quartiles we don't need to fully sort, just get the right value at the right place
  // this is O(n) instead of O(n log n) for full sorting, so a big difference for large vectors!
  // https://stackoverflow.com/questions/11964552/finding-quartiles
  auto const Q1 = vec.size() / 4;
  auto const Q2 = vec.size() / 2;
  auto const Q3 = Q1 + Q2;

  std::nth_element(vec.begin(), vec.begin() + Q1, vec.end());
  std::nth_element(vec.begin() + Q1 + 1, vec.begin() + Q2, vec.end());
  std::nth_element(vec.begin() + Q2 + 1, vec.begin() + Q3, vec.end());

  // minimum and maximum
  float min_value = *min_element(vec.begin(), vec.end());
  float max_value = *max_element(vec.begin(), vec.end());

  std::vector<float> out = {min_value, vec.at(Q1), vec.at(Q2), vec.at(Q3), max_value};
  return out;
}

void get_mean_std(std::vector<float> &v, float &mean, float &std) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  mean = sum / v.size();

  double sqsum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  std = std::sqrt(sqsum / v.size() - mean * mean);
}

void mc_simulations(std::atomic<int> &n_simulations,
                    const int max_n_simulations,
                    const int n_periods, const float initial_capital,
                    std::vector<float> &historical_returns,
                    std::vector<std::vector<float>> &mc_data,
                    std::vector<float> &final_values) {
  // TODO remove
  float monthly_return_mean = 7.0 / 12;
  float monthly_return_std = 10.0 / 12; // 68% is within 1 std from mean, 95% within 2 std, 99.7% within 3 std

  //pre-allocate so we can do parallel for
  final_values.reserve(max_n_simulations);
  mc_data.reserve(max_n_simulations);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for
  for (int n_sims = 0; n_sims < max_n_simulations; n_sims++) {
    std::vector<float> returns = sample_returns_historical(n_periods, historical_returns);
//    std::vector<float> returns = sample_returns_gaussian(n_months, monthly_return_mean, monthly_return_std);
    std::vector<float> values = many_updates(initial_capital, returns);
    final_values[n_sims] = values.back();
    mc_data[n_sims] = values;

    n_simulations++;
    if (n_simulations % 1000 == 0)
      fmt::print("{:d}/{:d} simulations done\n", n_simulations, max_n_simulations);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  fmt::print("All {} simulation done in {} s!\n", n_simulations, timediff / 1000.0);
}

int main(int, char **) {

  //-------------------------------------
  // Monte Carlo stock market simulations
  //-------------------------------------
  float initial_capital = 1000;

  int n_years = 30;
  int n_periods = 12 * n_years;

  // Monte Carlo
  std::atomic<int> n_simulations = 0; // atomic b/c shared between openMP threads
  int max_n_simulations = static_cast<int>(1e6);
  // limit max shown for plotting?
  int max_displayed_plots = 2000;

  // buffers to store results
  std::vector<std::vector<float>> mc_data(max_n_simulations, std::vector<float>(n_periods));
  std::vector<float> final_values(max_n_simulations);

  ////simulate by sampling from historical monthly returns
  std::string historical_returns_csv = "data/SP500_monthly_returns.csv";
  std::vector<float> historical_returns = read_historical_returns(historical_returns_csv);

  // MC sampling in background thread: https://hackernoon.com/learn-c-multi-threading-in-5-minutes-8b881c92941f
  std::thread t1(mc_simulations,
                 std::ref(n_simulations),
                 max_n_simulations,
                 n_periods,
                 initial_capital,
                 std::ref(historical_returns),
                 std::ref(mc_data),
                 std::ref(final_values));

  //-------------------------------------
  // GUI stuff
  //-------------------------------------
  glfwSetErrorCallback(glfw_error_callback);

  // Init to set up window
  if (!glfwInit())
    return 1;

  // Decide GL+GLSL versions
// Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1280, 720,
                                        "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
  if (window == nullptr)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

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
    glbinding::initialize([](const char *name) {
        return (glbinding::ProcAddress)glfwGetProcAddress(name);
    });
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader
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
  (void) io;

  // Setup Dear ImGui style
//  ImGui::StyleColorsDark();
  ImGui::StyleColorsClassic();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Our state
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // Main loop
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

    // horizontal line for desired final amount, and calculate how many simulations are below this
    static float min_final_amount;
    float max_final_value_simulations = *max_element(final_values.begin(), final_values.end());
    ImGui::SliderFloat("Desired final amount?", &min_final_amount, 0, max_final_value_simulations);

    int count_below_min = 0;
    for (auto value: final_values) {
      if (value < min_final_amount)
        count_below_min++;
    }

    if (ImPlot::BeginPlot("My Plot",
                          ImVec2(-1, height - 150), //leave some space for text below graph
                          ImPlotFlags_NoLegend)) {// | ImPlotAxisFlags_AutoFit)) {
      // plot config
      ImPlot::SetupAxes("Time (Months)", "Value", ImPlotAxisFlags_AutoFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);

      // TODO how to keep plot across frames and only plot new line?
      // plot all lines in PlotBuffer.
      // Limit number for performance & stop changing if all simulations are done
      for (int n_shown = 0; n_shown < max_displayed_plots; n_shown++) {
        if (n_shown >= mc_data.size())
          break;
        int idx = n_shown;
        if (n_simulations < max_n_simulations && mc_data.size() > max_displayed_plots)
          idx = int(rng_uniform(0, int(mc_data.size())));

        ImPlot::PlotLine(fmt::format("Simulation {}", idx).c_str(),
                         mc_data.at(idx).data(),
                         mc_data.at(idx).size());
      }

      // plot horizontal line of desired final amount
      std::vector<float> vec2(360, min_final_amount);
      ImPlot::PlotLine("MINIMUM", vec2.data(), vec2.size());

      ImPlot::EndPlot();
    }

    ImGui::Text("%s", fmt::format("#simulations: {:d}/{:d}", n_simulations, max_n_simulations).c_str());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);

    // calculate and print statistics
    std::vector<float> quartiles = quartile_sort(final_values);
    ImGui::Text("%s", fmt::format("min: {:.2f} | Q1: {:.2f} | median: {:.2f} | Q3: {:.2f} | max: {:.2f}",
                                  quartiles.at(0),
                                  quartiles.at(1),
                                  quartiles.at(2),
                                  quartiles.at(3),
                                  quartiles.at(4)).c_str());

    float mean, std;
    get_mean_std(final_values, mean, std);
    ImGui::Text("%s", fmt::format("mean: {:.2f} | std: {:.2f}", mean, std).c_str());

    ImGui::Text("%s", fmt::format("{:d}/{:d} ({:.4f}%) are below target final value",
                                  count_below_min, n_simulations,
                                  100 * float(count_below_min) / n_simulations).c_str());
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

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
