import cv2

# Load an image
image = cv2.imread('fruit_samp.png')

# Example bounding box data: [x, y, width, height]
bbox = [50, 50, 150, 100]
label = "Apple: Fresh"

# Draw bounding box and label
cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()