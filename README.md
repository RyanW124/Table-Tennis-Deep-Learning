# Table Tennis Deep Learning
 Classify top spin and back spin shots

## Procedure

### 1. Determine when a shot has been made

This is done so that every time the ball is hit, the segment of video is trimmed and fed to the model to classify the type of shot

At first, I used a CNN to identify the position of the ball and determine when the x position of the ball changes direction. This didn't work out because the ball is too small and CNN loses a lot of its information. Currently, I'm planning on using a CNN just on the few columns of pixel at the middle of the video to determine whether a ball passed through. This should work since reducing the size of data allows me to upscale the image whilst being able to train the model. 

### 2. Manually label each video segment

### 3. Train model to classify shot type

Since the video segments vary in length and have high dimensions, a CNN-LSTM architecture would be used. 

