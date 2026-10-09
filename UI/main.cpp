#include "MainWindow.h"

#include <QApplication>
#include <QTimer>
#include <QTranslator>
#include <iostream>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("Shape Match Qt Client");
    const bool smokeTest = app.arguments().contains(QStringLiteral("--smoke-test"));
    if (smokeTest) {
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                          QCoreApplication::applicationDirPath() + QStringLiteral("/smoke-test-settings"));
        QCoreApplication::setOrganizationName(QStringLiteral("edgeMatchingSmokeTest"));
        QTranslator catalog;
        if (!catalog.load(QStringLiteral("shape_match_en.qm"), QCoreApplication::applicationDirPath())) {
            std::cerr << "Translation catalog could not be loaded.\n";
            return 1;
        }
    }
    MainWindow window;
    window.show();
    if (smokeTest) {
        window.selectLanguage(true);
        window.selectLanguage(false);
        QTimer::singleShot(250, &app, &QCoreApplication::quit);
        std::cout << "Qt " << qVersion() << " startup and translation switching succeeded.\n";
    }
    return app.exec();
}
