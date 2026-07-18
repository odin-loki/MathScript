#include <QApplication>
#include <QDir>

#include "gui/MainWindow.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("MathScript");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("MathScript");
    app.setStyle("Fusion");

    MainWindow window;
    window.show();
    return app.exec();
}
