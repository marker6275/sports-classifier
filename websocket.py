import json
import threading
import asyncio
import websockets

class WebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.running = False
        
    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"WebSocket client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        self.clients.discard(websocket)
        print(f"WebSocket client disconnected. Total clients: {len(self.clients)}")
        
    async def handler(self, websocket):
        await self.register_client(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self.unregister_client(websocket)
            
    async def broadcast(self, message):
        if self.clients:
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            self.clients -= disconnected
            
    def send_update(self, data):
        if self.running and self.clients:
            message = json.dumps(data)
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
            
    def start(self):
        async def run_server():
            async with websockets.serve(self.handler, self.host, self.port):
                self.running = True
                if self.host == '0.0.0.0':
                    print(f"WebSocket server started on ws://localhost:{self.port} (accessible from all network interfaces)")
                else:
                    print(f"WebSocket server started on ws://{self.host}:{self.port}")
                await asyncio.Future()
                
        self.loop = asyncio.new_event_loop()
        thread = threading.Thread(target=lambda: self.loop.run_until_complete(run_server()), daemon=True)
        thread.start()
        return thread
        
    def stop(self):
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

