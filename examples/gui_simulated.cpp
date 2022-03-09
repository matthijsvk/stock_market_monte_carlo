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
#include "implot_demo.cpp"
#include <fmt/core.h>

#include <simulations.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>

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

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

struct Data {
    std::vector<double> x_axis, y_axis;
};

double rng() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, 1);
    return dist(e2);
}

// utility structure for realtime plot
struct PlotBuffer {
    int MaxSize;
    int Offset;
    std::vector<std::vector<float>> Data;

    explicit PlotBuffer(int max_size = 2000) {
        MaxSize = max_size;
        Offset = 0;
    }

    void AddLine(std::vector<float> &ys) {
        if (Data.size() < MaxSize)
            Data.push_back(ys);
        else {
            Data[Offset] = ys;
            Offset = (Offset + 1) % MaxSize;
        }
    }

    void Erase() {
        if (Data.size() > 0) {
            Data.pop_back();
            Offset = 0;
        }
    }
};

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

    std::nth_element(vec.begin(), vec.begin(), vec.end());
    std::nth_element(vec.begin(), vec.end(), vec.end());

    std::vector<float> out = {vec.at(0), vec.at(Q1), vec.at(Q2), vec.at(Q3), vec.back()};
    return out;
}

void get_mean_std(std::vector<float> &v, float &mean, float &std) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    mean = sum / v.size();

    double sqsum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    std = std::sqrt(sqsum / v.size() - mean * mean);
}

int main(int, char **) {
    glfwSetErrorCallback(glfw_error_callback);

    // Init to setup window
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
                                          "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
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
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
    // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
    // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can
    // also load multiple fonts and use ImGui::PushFont()/PopFont() to select
    // them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you
    // need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please
    // handle those errors in your application (e.g. use an assertion, or
    // display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and
    // stored into a texture when calling
    // ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame
    // below will call.
    // - Read 'docs/FONTS.txt' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string
    // literal you need to write a double backslash \\ !
    // io.Fonts->AddFontDefault();
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    // ImFont* font =
    // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f,
    // NULL, io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != NULL);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Monte Carlo stock market
    // Fund and Market configuration
    ///////////////////////////////////////////
    float initial_capital = 1000;

    float monthly_return_mean = 6.0 / 12;
    float monthly_return_std = 12.0 / 12;  // 68% is within 1 std from mean, 95% within 2 std, 99.7% within 3 std

    int n_years = 30;
    int n_months = 12 * n_years;

    // Monte Carlo
    int n_simulations = 0;
    int max_n_simulations = 1000;

    static PlotBuffer mc_data; //stores finished simulations, used for plotting

    ////simulate by sampling from historical monthly returns
    std::string historical_returns_csv = "data/SP500_monthly_returns.csv";
    std::vector<float> historical_returns = read_historical_returns(historical_returns_csv);

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
        // right now we just do 1 per frame, so 60 simulations/second
        // we could do way more using decoupling plotting & simulation using separate threads...
        std::vector<float> returns;
        std::vector<float> values;
        if (n_simulations < max_n_simulations) {
            returns = sample_returns_historical(n_months, historical_returns);
//            returns = sample_returns_gaussian(n_months, monthly_return_mean, monthly_return_std);
            values = many_updates(initial_capital, returns);
            mc_data.AddLine(values);
            n_simulations++;
        }

        // show horizontal line for desired final amount
        static float min_final_amount;
        ImGui::SliderFloat("Desired final amount?", &min_final_amount, 0, 10000);
        // calculate fraction of final values that are below the user-set minimum final value
        int count_below_min = 0;

        // keep stats
        std::vector<float> final_values = {};
        if (ImPlot::BeginPlot("My Plot",
                              ImVec2(-1, height - 150), //leave some space for text below graph
                              ImPlotFlags_NoLegend)) {// | ImPlotAxisFlags_AutoFit)) {
            // plot config
            ImPlot::SetupAxes("Time (Months)", "Value", ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
            // todo how to set minimum but leave maximum automatic?
//            ImPlot::SetupAxisLimits(ImAxis_Y1, -100, NULL);

            // TODO how to keep plot across frames and only plot new line?
            // plot all lines in PlotBuffer
            int idx = 0;
            for (auto &vec: mc_data.Data) {
                std::string name = fmt::format("Simulation {}", idx);
                ImPlot::PlotLine(name.c_str(), vec.data(), vec.size());
                final_values.push_back(vec.back());
                if (vec.back() < min_final_amount)
                    count_below_min++;
                idx++;
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
                                      quartiles.at(0), quartiles.at(1), quartiles.at(2), quartiles.at(3),
                                      quartiles.at(4)).c_str());

        float mean, std;
        get_mean_std(final_values, mean, std);
        ImGui::Text("%s", fmt::format("mean: {:.2f} | std: {:.2f}", mean, std).c_str());


        ImGui::Text("%s", fmt::format("{:d}/{:d} ({:.2f}%) are below target final value",
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
