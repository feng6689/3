import cv2
import numpy as np

def detect_shapes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    triangle_count = 0
    circle_count = 0
    rectangle_count = 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        
        x, y, w, h = cv2.boundingRect(contour)
        
        if vertices == 3:
            triangle_count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Triangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        elif vertices == 4:
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                continue
            else:
                rectangle_count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        else:
            area_contour = area
            center_x, center_y = x + w // 2, y + h // 2
            radius = max(w, h) // 2
            circle_area = np.pi * radius * radius
            ratio = area_contour / circle_area
            
            if 0.7 <= ratio <= 1.3:
                circle_count += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Circle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    print(f"识别到的形状数量:")
    print(f"圆形: {circle_count} 个")
    print(f"三角形: {triangle_count} 个")
    print(f"矩形: {rectangle_count} 个")
    
    output_path = 'result.jpg'
    cv2.imwrite(output_path, img)
    print(f"\n标注后的图片已保存为: {output_path}")
    
    cv2.imshow('Shape Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_shapes('1.jpg')


