import sys
import cv2
import time
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder

from vehicle_detection import VehicleDetection

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file of the object Detection model.")
   
                       
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CAM or path to image or video file.")
    

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
                       
    return parser

def handle_input_type(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        input_type = 'cam'
        
    # Check if the input is a video    
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.error('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')    
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
    # Initialise the classes
    
    try:
        vehicle_detection = VehicleDetection(args.vd_model, args.device)
        
    except:
        log.error('Please enter a valid model file address')    
        sys.exit()

    

    start_load = time.time()
    
    # Load the models 
    vehicle_detection.load_model()
    log.debug("Models loaded: time: {:.3f} ms".format((time.time() - start_load) * 1000))
    end_load = time.time() -  start_load 
    
    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Load the video capture
    feed.load_data()
    frame_count = 0
    start_inf = time.time()
    
    # Read from the video capture 
    for flag, frame in feed.next_batch():
        if not flag:
            break
        frame_count += 1
        try:
            # Run inference on the models     
            vehicle = vehicle_detection.predict(frame)
            ## If no face detected move back to the top of the loop
                                    
            
            # cv2.moveWindow('Computer pointer control',  10,10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("User stops the program")
                break
            #mouse_controller.move(gaze_vector[0], gaze_vector[1])

        except Exception as e:
            log.warning(str(e) + " for frame " + str(frame_count))
            continue
        # Display the resulting frame
        
     
    end_inf = time.time() - start_inf
    log.info("\nTotal loading time: {}\nTotal inference time: {}\nFPS: {}".format(end_load, end_inf,frame_count/end_inf))
    
    # Release the capture
    feed.close()
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
    log.debug("The program debug successfully")

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    #time.sleep(7)
    # Grab command line args
    args = build_argparser().parse_args()

    #Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()
