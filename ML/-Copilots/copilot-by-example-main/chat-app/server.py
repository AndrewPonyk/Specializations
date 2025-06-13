import socket
import threading

connected_clients = []

def handle_client(client_socket):
    connected_clients.append(client_socket)

    while True:
        # Receive data from the client
        message = client_socket.recv(1024)

        if not message:
            # The client has disconnected
            break

        # Here you can add code to handle the received message
        decoded_message = message.decode('utf-8')
        print(f"Received message: {decoded_message}")

        # Forward the message to all other connected clients
        for client in connected_clients:
            if client != client_socket:
                client.sendall(message)

    # Remove the client from the list of connected clients
    connected_clients.remove(client_socket)

    # Close the client socket
    client_socket.close()

def start_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_address = ('localhost', 3000)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(5)
    print('Server listening on {}:{}'.format(*server_address))

    while True:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print('New connection from {}:{}'.format(*client_address))

        # Create a new thread to handle the client connection
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

start_server()