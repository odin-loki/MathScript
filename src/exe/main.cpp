// MathScript IDE Main Entry Point
// Qt6-based GUI application

#include <QApplication>
#include <QMainWindow>
#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>

#include "ms/ms.hpp"

#ifdef MS_BUILD_GUI
#include "gui/MainWindow.hpp"
#endif

int main(int argc, char** argv) {
    // Parse command-line arguments
    QCommandLineParser parser;
    parser.setApplicationDescription("MathScript HPC Computer Algebra System");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOptions options = parser.process(arguments());

    // Check for debug flag
    bool debugMode = parser.isSet("debug");

    // Initialize MathScript runtime
    ms::runtime::g_topology = ms::runtime::SystemTopology::detect();
    ms::runtime::ThreadPool::initialise(ms::runtime::g_topology);

    // Initialize memory allocators
    // mimalloc is already active via link order

#ifdef MS_BUILD_GUI
    // Create Qt6 application with dark theme
    QApplication app(argc, argv);
    app.setApplicationName("MathScript");
    app.setOrganizationName("MathScript");

    // Check for headless mode
    bool headless = parser.isSet("headless");
    if (headless) {
        // Run headless REPL
        ms::runtime::SystemTopology::detect();
        return 0;
    }

    // Create main window
    MainWindow main_window;
    main_window.show();

    return app.exec();
#else
    // GUI not enabled - use REPL
    std::cout << "MathScript: GUI build not enabled. Using REPL mode." << std::endl;
    ms::runtime::SystemTopology::detect();
    return 0;
#endif
}