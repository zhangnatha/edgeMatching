#include "ImageView.h"

#include <QGraphicsPixmapItem>
#include <QLineF>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QScrollBar>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

ImageView::ImageView(QWidget* parent)
    : QGraphicsView(parent), scene_(new QGraphicsScene(this)),
      pixmap_(new QGraphicsPixmapItem()), zoom_(1.0)
{
    scene_->addItem(pixmap_);
    setScene(scene_);
    setBackgroundBrush(QColor(30, 34, 42));
    setFrameShape(QFrame::NoFrame);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::NoAnchor);
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setMinimumSize(260, 220);
}

void ImageView::setImage(const QImage& image)
{
    image_ = image;
    overlays_.clear();
    pixmap_->setPixmap(QPixmap::fromImage(image_));
    scene_->setSceneRect(pixmap_->boundingRect());
    updatePixelRendering();
    fitImage();
}

void ImageView::setOverlays(const QVector<OverlayPath>& overlays)
{
    overlays_ = overlays;
    viewport()->update();
}

void ImageView::clearOverlays()
{
    if (overlays_.isEmpty()) return;
    overlays_.clear();
    viewport()->update();
}

void ImageView::setContours(const QVector<ContourPath>& contours)
{
    setOverlays(contours);
}

void ImageView::clearContours()
{
    clearOverlays();
}

void ImageView::fitImage()
{
    if (pixmap_->pixmap().isNull()) return;
    resetTransform();
    fitInView(pixmap_, Qt::KeepAspectRatio);
    zoom_ = transform().m11();
    updatePixelRendering();
    viewport()->update();
}

void ImageView::resetZoom()
{
    if (pixmap_->pixmap().isNull()) return;
    resetTransform();
    zoom_ = 1.0;
    const QPointF center = pixmap_->boundingRect().center();
    centerOn(center);
    updatePixelRendering();
    viewport()->update();
}

void ImageView::wheelEvent(QWheelEvent* event)
{
    if (pixmap_->pixmap().isNull()) { event->ignore(); return; }
    const QPointF sceneBefore = mapToScene(event->pos());
    const double factor = event->angleDelta().y() > 0 ? 1.20 : (1.0 / 1.20);
    const double next = zoom_ * factor;
    if (next < 0.05 || next > 40.0) { event->accept(); return; }
    scale(factor, factor);
    zoom_ = next;
    const QPointF sceneAfter = mapToScene(event->pos());
    const QPointF delta = sceneAfter - sceneBefore;
    translate(delta.x(), delta.y());
    updatePixelRendering();
    viewport()->update();
    event->accept();
}

void ImageView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    if (!image_.isNull() && qFuzzyCompare(zoom_, 1.0)) fitImage();
    viewport()->update();
}

void ImageView::updatePixelRendering()
{
    // At this scale each source pixel is large enough to inspect individually;
    // interpolation would blur the pixel blocks beneath the grid.
    setRenderHint(QPainter::SmoothPixmapTransform, zoom_ < 8.0);
}

void ImageView::drawForeground(QPainter* painter, const QRectF& rect)
{
    QGraphicsView::drawForeground(painter, rect);
    if (pixmap_->pixmap().isNull() || image_.isNull()) return;

    const QRectF imageRect = pixmap_->boundingRect();
    const QRectF visible = rect.intersected(imageRect);

    // Pixel grid is part of the raster layer.  It is deliberately disabled
    // for normal zoom and pixelated at high zoom, before the vector overlay.
    if (zoom_ >= 8.0 && !visible.isEmpty()) {
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, false);
        QPen gridPen(QColor(105, 105, 105));
        gridPen.setWidth(0); // cosmetic: one device pixel at every zoom level
        painter->setPen(gridPen);

        const int x0 = std::max(0, static_cast<int>(std::floor(visible.left())));
        const int x1 = std::min(image_.width(),
                                static_cast<int>(std::ceil(visible.right())));
        const int y0 = std::max(0, static_cast<int>(std::floor(visible.top())));
        const int y1 = std::min(image_.height(),
                                static_cast<int>(std::ceil(visible.bottom())));
        for (int x = x0; x <= x1; ++x)
            painter->drawLine(QPointF(x, visible.top()), QPointF(x, visible.bottom()));
        for (int y = y0; y <= y1; ++y)
            painter->drawLine(QPointF(visible.left(), y), QPointF(visible.right(), y));
        painter->restore();
    }

    if (overlays_.isEmpty()) return;

    // Draw fractional scene-space contours independently from the pixmap.
    // Cosmetic width keeps the overlay a clean approximately one-device-pixel
    // line, even when the image underneath is magnified with nearest-neighbor.
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);
    painter->setBrush(Qt::NoBrush);
    for (const OverlayPath& overlay : overlays_) {
        if (overlay.points.size() < 2) continue;
        QPen pen(overlay.color);
        pen.setWidthF(overlay.width);
        pen.setCosmetic(overlay.cosmetic);
        painter->setPen(pen);
        QPainterPath path;
        path.moveTo(overlay.points.front());
        for (int i = 1; i < overlay.points.size(); ++i)
            path.lineTo(overlay.points.at(i));
        if (overlay.closed) path.closeSubpath();
        painter->drawPath(path);
        if (!overlay.label.isEmpty()) {
            painter->setPen(QPen(overlay.color));
            QFont font = painter->font(); font.setBold(true); font.setPointSizeF(11.0);
            painter->setFont(font);
            painter->drawText(overlay.labelPosition, overlay.label);
        }

        if (overlay.arrow) {
            const QPointF tip = overlay.points.back();
            const QPointF tail = overlay.points.at(overlay.points.size() - 2);
            const QLineF direction(tail, tip);
            if (direction.length() > 1e-6) {
                const double angle = std::atan2(direction.dy(), direction.dx());
                const double size = std::max<qreal>(1.0, overlay.arrowSize);
                const QPointF left = tip - QPointF(size * std::cos(angle - M_PI / 6.0),
                                                    size * std::sin(angle - M_PI / 6.0));
                const QPointF right = tip - QPointF(size * std::cos(angle + M_PI / 6.0),
                                                     size * std::sin(angle + M_PI / 6.0));
                painter->drawLine(tip, left);
                painter->drawLine(tip, right);
            }
        }
    }
    painter->restore();
}
