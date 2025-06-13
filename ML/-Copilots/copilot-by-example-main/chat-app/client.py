import socket
import threading

# Create a client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server socket on port 3000
server_address = ('localhost', 3000)
client_socket.connect(server_address)

def send_message():
    while True:
        message = input('Enter a message: ')
        client_socket.send(message.encode())

def receive_message():
    while True:
        response = client_socket.recv(1024).decode()
        print('Received:', response)

# Create two threads for sending and receiving messages
send_thread = threading.Thread(target=send_message)
receive_thread = threading.Thread(target=receive_message)

# Start both threads
send_thread.start()
receive_thread.start()

# Wait for both threads to finish
send_thread.join()
receive_thread.join()

# Close the client socket when you're done
client_socket.close()