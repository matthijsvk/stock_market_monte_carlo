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
#define GLFW_INCLUDE_NONE  // GLFW including OpenGL headers causes ambiguity or
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

unsigned TausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}
unsigned LCGStep(unsigned int &z, unsigned int A, unsigned int C) {
  return z = (A * z + C);
}
float HybridTaus(unsigned int &z1,
                 unsigned int &z2,
                 unsigned int &z3,
                 unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
      TausStep(z2, 2, 25, 4, 4294967288UL) ^
      TausStep(z3, 3, 11, 17, 4294967280UL) ^
      LCGStep(z4, 1664525, 1013904223UL));
}

float HybridTausSimple(unsigned int &z1,
                       unsigned int &z2,
                       unsigned int &z3,
                       unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
      TausStep(z2, 2, 25, 4, 4294967288UL));
}

float HybridTausMedium(unsigned int &z1,
                 unsigned int &z2,
                 unsigned int &z3,
                 unsigned int &z4) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  return float(2.3283064365387e-10) * (TausStep(z1, 13, 19, 12, 4294967294UL) ^
      TausStep(z2, 2, 25, 4, 4294967288UL) ^
      TausStep(z3, 3, 11, 17, 4294967280UL));
}

std::vector<float> test_RNG(int n) {
//  unsigned int rstate[4];
//  for (int i = 0; i < 4; i++) rstate[i] = i * 12371;
  std::vector<float> values(n, 0);
  unsigned int rstate[] = {1, 21701, 1297, 65537};

  for (int i = 0; i < n; i++) {
//    float val = HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);
//    float val = HybridTausMedium(rstate[0], rstate[1], rstate[2], rstate[3]);
    float val = HybridTausSimple(rstate[0], rstate[1], rstate[2], rstate[3]);
    printf("%f\t", val);
    values[n] = val;
  }
  return values;
}

int main(int argc, char *argv[]) {
  fmt::print("argc: {}\n", argc);
  int n;
  if (argc == 2) {
    char *end;
    n = int(std::strtol(argv[1], &end, 10));
    fmt::print("n: {}\n", n);
  } else {
    fmt::print(
        "usage: visualize_rng <n>>");
    exit(0);
  }
  // TODO preallocate?
//  std::vector<float> values(n, 0);

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
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Required on Mac
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
  GLFWwindow *window = glfwCreateWindow(window_width,
                                        window_height,
                                        "RNG distribution (1D)",
                                        nullptr,
                                        nullptr);
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
  glbinding::initialize([](const char *name) {
    return (glbinding::ProcAddress)glfwGetProcAddress(name);
  });
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
  std::vector<float> values;
  int maxval = 1000;
  std::vector<int> counts(maxval, 0);
  unsigned int rstate[] = {0, 21701, 1297, 65537};
  int prev_n = 0;
  bool pause = 0;
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
    ImGui::Begin("RNG distribution");

    // horizontal line for desired final amount, and to calculate how many
    // simulations are below this
    ImGui::Indent(0.25 * window_width);
    ImGui::SetNextItemWidth(0.5 * window_width);
    ImGui::SliderInt("N values?", &n, 0, 100000);
    if (prev_n != n) {
      pause = 0;
      prev_n = n;
    }
    ImGui::Unindent(0.25 * window_width);

    // recompute
    ImGui::Indent(0.5 * window_width);
    ImGui::Checkbox("Pause?", &pause);
    ImGui::Unindent(0.5 * window_width);
    if (!pause) {
      for (uint i = 0; i < counts.size(); i++) {
        counts[i] = 0;
      }
      for (uint i = 0; i < n; i++) {
        unsigned int idx = maxval * HybridTaus(rstate[0], rstate[1], rstate[2], rstate[3]);
        counts[idx] += 1;
      }
    }

    // Plot
    if (ImPlot::BeginPlot("My Plot", ImVec2(-1, height - 100), ImPlotAxisFlags_AutoFit)) {
      ImPlot::SetupAxes("idx", "count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
//      ImPlot::PlotLine(fmt::format("Histogram").c_str(), counts.data(), counts.size());
      ImPlot::PlotBars(fmt::format("Histogram").c_str(), counts.data(), counts.size(), 0.67);
    }
    ImPlot::EndPlot();

    ImGui::Text("%s", fmt::format("#points: {:d}", n).c_str());
    ImGui::Text("Application average %.1f FPS", ImGui::GetIO().Framerate);

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

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

