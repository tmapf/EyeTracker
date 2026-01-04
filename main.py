import cv2

cap = cv2.VideoCapture(0)


def main():
    while True:
        ret, frame = cap.read()
        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    cap.release()
    cv2.destroyAllWindows()


