import subprocess
import time
import os
import smtplib
import base64
import pickle
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import argparse

PORT = 5001

### PATH MANAGEMENT ###
BASE_PROJECT_PATH = "/home/yvonne/Documents/CNN_compact/"
CLIENT_BASE_PATH = '/home/test/Documents/CNN/' 
use_iot = False
SERVER_SCRIPT = BASE_PROJECT_PATH + "RIC/Server.py"
CLIENT_SCRIPT = BASE_PROJECT_PATH + "RIC/IoT_client.py"

output_base_dir = os.path.join(BASE_PROJECT_PATH, "output/", "result2/")
outputname = ""
SERVER_LOG = ""
CLIENT_LOG = ""
IMAGE_DIR = ""

IMAGES_TO_ATTACH = [
    "0.png",
    "1.png",
    "2.png"
]
### PATH MANAGEMENT END ###

# Server-Parameters
def generate_args(mode, sal_for_he, compress, sal_for_cs, encrypt, share_guide):
    """
    Generates a list of arguments for the Server or Client script based on the provided flags.
    """
    args = [f'--base_project_path={BASE_PROJECT_PATH}']
    if mode == 'train':
        args.append('--train')
    elif mode == 'test':
        args.append('--test')
        
    if sal_for_he:
        args.append('--sal_for_he')
    if compress:
        args.append('--compress')
    if sal_for_cs:
        args.append('--sal_for_cs')
    if encrypt:
        args.append('--encrypt')
    if share_guide:
        args.append('--share_guide')
    
    return args

# E-Mail
SENDER_EMAIL = "xxanyuxx@gmail.com" 
TOKEN_DIR = os.path.join(BASE_PROJECT_PATH, ".credentials")
os.makedirs(TOKEN_DIR, exist_ok=True)
TOKEN_FILE = os.path.join(TOKEN_DIR, 'token.pickle')    
RECEIVER_EMAIL = "ygross_99@gmx.de" 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def kill_process_on_port(port):
    """
    Findet und beendet den Prozess, der den angegebenen Port verwendet.
    """
    print(f"Suche nach blockierendem Prozess auf Port {port}...")
    try:
        output = subprocess.run(
            ['sudo', 'lsof', '-t', '-i', f':{port}'],
            capture_output=True, text=True, check=True
        )
        pids = output.stdout.strip().split('\n')
        
        if pids == ['']:
            print(f"Kein Prozess gefunden, der Port {port} verwendet.")
            return

        for pid in pids:
            if pid:
                print(f"Prozess mit PID {pid} blockiert Port {port}. Versuche zu beenden...")
                try:
                    subprocess.run(['sudo', 'kill', '-9', pid], check=True)
                    print(f"Prozess {pid} erfolgreich beendet.")
                except subprocess.CalledProcessError as e:
                    print(f"Fehler: Konnte Prozess {pid} nicht beenden. Möglicherweise keine Berechtigung: {e}")
                except FileNotFoundError:
                    print("Fehler: 'kill' Befehl nicht gefunden. Ist er in Ihrem PATH?")
    
    except subprocess.CalledProcessError as e:
        print(f"Fehler: 'lsof' Befehl fehlgeschlagen. Möglicherweise keine Berechtigung oder 'lsof' ist nicht installiert: {e}")
    except FileNotFoundError:
        print("Fehler: 'lsof' Befehl nicht gefunden. Bitte installieren Sie es oder prüfen Sie Ihren PATH.")


def run_script(script_path, args, log_file, wait_for_ready=False):  
    print(f"Starting '{script_path}' with Arguments: {args}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(['python3', script_path] + args, stdout=f, stderr=f, text=True)
    
    if wait_for_ready:
        print("Wait 100 Sekonds, until Server is ready...")
        time.sleep(50) 
    
    return process

def get_last_lines(log_file, num_lines=50):
    if not os.path.exists(log_file):
        return f"Log File '{log_file}' not found."
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        return "".join(lines[-num_lines:])

def send_email_with_logs_and_images(server_log_content, client_log_content, image_paths, args):
    
    print("Sending Mail...")
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        print("Error: The Token file does not exist or is damaged. Please regenerate.")
        return

    try:
        service = build('gmail', 'v1', credentials=creds)

        message = MIMEMultipart()
        message['to'] = RECEIVER_EMAIL
        message['from'] = SENDER_EMAIL
        message['subject'] = "Modell-Training- and Testreview"

        body = f"Training and Test for {args} are done. Excerpt of Log Data as follows:\n\n"
        body += "--- SERVER LOG ---\n"
        body += server_log_content
        body += "\n\n--- CLIENT LOG ---\n"
        body += client_log_content

        message.attach(MIMEText(body, 'plain'))

        for img_path in image_paths:
            if os.path.exists(img_path):
                try:
                    part = MIMEBase('application', 'octet-stream')
                    with open(img_path, 'rb') as attachment:
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition',
                                    f"attachment; filename= {os.path.basename(img_path)}")
                    message.attach(part)
                except Exception as e:
                    print(f"Error attaching Image {img_path}: {e}")
            else:
                print(f"Warning: Image '{img_path}' not found.")

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        send_message = {'raw': raw_message}
        (service.users().messages().send(userId='me', body=send_message).execute())
        
        print("E-Mail send successfully.")
    except Exception as e:
        print(f"Error sending Mail: {e}")

def pipe(trargs, teargs, train = True):
    kill_process_on_port(PORT)
    if train:
        # Step 1: Train script
        train_process = run_script(SERVER_SCRIPT, trargs, SERVER_LOG)
        train_process.wait() 
        print("Training completed.")

    # Step 2: Server in test modus and wait for client
    test_process = run_script(SERVER_SCRIPT, teargs, SERVER_LOG, wait_for_ready=True)

    # Step 3: start client
    if use_iot:
        # Step 3: start client via SSH
        ssh_command = f'ssh test@192.168.55.1 "cd Documents && python3 IoT_client.py --base_project_path={CLIENT_BASE_PATH}"'
        print(f"Starting client via SSH: {ssh_command}")
        with open(CLIENT_LOG, "w") as f:
            client_process = subprocess.Popen(ssh_command, shell=True, stdout=f, stderr=f, text=True)
        client_process.wait() # Warten, bis der Client beendet ist
        #test_process.terminate() #Not needed as Server stops automatically
        print("Test completed.")
    else:
        client_process = run_script(CLIENT_SCRIPT, [], CLIENT_LOG)
        client_process.wait() # Warten, bis der Client beendet ist
        #test_process.terminate() #Not needed as Server stops automatically
        print("Test completed.")

    '''# Step 4: read logs
    server_log_content = get_last_lines(SERVER_LOG)
    client_log_content = get_last_lines(CLIENT_LOG)

    # Step 5: send Mail with logs and iamges
    image_paths_to_attach = [os.path.join(IMAGE_DIR, img_name) for img_name in IMAGES_TO_ATTACH]
    send_email_with_logs_and_images(server_log_content, client_log_content, image_paths_to_attach, teargs)
    '''
    print("Pipeline concluded.")

def run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=True):
    """
    Helper function to set up and run a single pipeline configuration.
    """
    global outputname, SERVER_LOG, CLIENT_LOG, IMAGE_DIR

    # Derived flags logic
    color = False
    if compress and not encrypt:
        color = False
    else:
        color = True

    svhn = False
    if not compress and encrypt:
        svhn = True
    else:
        svhn = False
    
    # Build the output directory name
    dataset_name_segment = "svhn" if svhn else "stl10"
    sal_for_he_segment = "sal_he" if sal_for_he else "no_sal_he"
    sal_for_cs_segment = "sal_cs" if sal_for_cs else "no_sal_cs"
    compress_segment = "compressed" if compress else "uncompressed"
    encrypt_segment = "encrypted" if encrypt else "unencrypted"
    share_guide_segment = "share_guide" if share_guide else "no_share_guide"
    color_segment = "color" if color else "grayscale"

    output_dir_name = (
        f"{dataset_name_segment}_"
        f"{sal_for_he_segment}_"
        f"{sal_for_cs_segment}_"
        f"{compress_segment}_"
        f"{encrypt_segment}_"
        f"{share_guide_segment}_"
        f"{color_segment}/"
    )
    
    # Redefine the global variables for the current run
    outputname = os.path.join(output_base_dir, output_dir_name)
    os.makedirs(outputname, exist_ok=True)
    print(f"\n--- Starting {'Training and ' if train_mode else ''}Testing with Flags: {output_dir_name.strip('/')} ---")
    print(f"Output directory (Old files of the same flags will be overwritten!): {outputname}")

    SERVER_LOG = outputname + "server_log.txt"
    CLIENT_LOG = outputname + "client_log.txt"
    IMAGE_DIR = outputname + "testimages"

    # Generate arguments
    TRAIN_ARGS = generate_args('train', sal_for_he, compress, sal_for_cs, encrypt, share_guide)
    TEST_ARGS = generate_args('test', sal_for_he, compress, sal_for_cs, encrypt, share_guide)

    # Run the pipeline
    pipe(TRAIN_ARGS, TEST_ARGS, train=train_mode)

if __name__ == "__main__":
    '''
    #encrypt feat
    sal_for_he = False
    compress = False
    sal_for_cs = False
    encrypt = True
    share_guide = False
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)

    #encrypt sal
    sal_for_he = True
    compress = False
    sal_for_cs = False
    encrypt = True
    share_guide = False
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)

    
    #compress feat
    sal_for_he = False
    compress = True
    sal_for_cs = False
    encrypt = False
    share_guide = False
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)

    #compress sal
    sal_for_he = False
    compress = True
    sal_for_cs = True
    encrypt = False
    share_guide = False
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)
    
    #pipe unshared
    sal_for_he = False
    compress = True
    sal_for_cs = True
    encrypt = True
    share_guide = False
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)
'''
    #pipe share feat
    sal_for_he = False
    compress = True
    sal_for_cs = False
    encrypt = True
    share_guide = True
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)
'''
    #pipe share sal
    sal_for_he = True
    compress = True
    sal_for_cs = True
    encrypt = True
    share_guide = True
    
    # Run the full pipeline (train and test) with the initial configuration
    run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)
    '''