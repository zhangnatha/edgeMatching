#include "MainWindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("Shape Match Qt Client");
    MainWindow window;
    window.show();
    return app.exec();
}
