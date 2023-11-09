import dlib
import cv2


def detect_faces_from_path(path, debug=False):
    image = cv2.imread(path)
    detect_faces(image, debug)


def detect_faces(image, debug=False):
    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if debug:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)

    result = []
    for face in faces:
        result.append([face.left(), face.top(), face.width(), face.height()])
    return result


if __name__ == "__main__":
    detect_faces_from_path("images/nadia.png", True)
    detect_faces_from_path("images/george.JPG", False)
    detect_faces_from_path("images/group.JPG", True)
    cv2.destroyAllWindows()
