#pragma once

#include <QGraphicsView>
#include <QColor>
#include <QImage>
#include <QPolygonF>
#include <QVector>

class QPainter;

class ImageView : public QGraphicsView
{
    Q_OBJECT
public:
    struct OverlayPath
    {
        QPolygonF points;
        bool closed = false;
        QColor color = QColor(0, 255, 0);
        qreal width = 1.0;
        bool cosmetic = true;
        bool arrow = false;
        qreal arrowSize = 8.0;
        QString label;
        QPointF labelPosition;
    };
    using ContourPath = OverlayPath;

    explicit ImageView(QWidget* parent = nullptr);

public slots:
    void setImage(const QImage& image);
    void setOverlays(const QVector<OverlayPath>& overlays);
    void clearOverlays();
    void setContours(const QVector<ContourPath>& contours);
    void clearContours();
    void fitImage();
    void resetZoom();

protected:
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void drawForeground(QPainter* painter, const QRectF& rect) override;

private:
    void updatePixelRendering();

    QGraphicsScene* scene_;
    QGraphicsPixmapItem* pixmap_;
    QImage image_;
    QVector<OverlayPath> overlays_;
    double zoom_;
};
