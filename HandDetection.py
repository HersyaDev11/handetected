import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

                                                                                                               
with mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,             
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5    
) as hands:
                                 
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera.")
            break

        
        image = cv2.flip(image, 1)
        
       
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
       
        results = hands.process(image_rgb)
        
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        
        cv2.imshow("Hand Detection", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
