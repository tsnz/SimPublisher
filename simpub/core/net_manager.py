import concurrent.futures
import enum
from typing import List, Dict, Optional
from typing import Callable, TypedDict, Union
import asyncio
from asyncio import sleep as async_sleep
import zmq
import zmq.asyncio
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
import time
import json
from json import dumps
import uuid
import abc
import concurrent

from .log import logger

IPAddress = str
TopicName = str
ServiceName = str
AsyncSocket = zmq.asyncio.Socket


class ServerPort(int, enum.Enum):
    # ServerPort and ClientPort need using .value to get the port number
    # which is not supposed to be used in this way
    DISCOVERY = 7720
    SERVICE = 7721
    TOPIC = 7722


class HostInfo(TypedDict):
    name: str
    instance: str  # hash code
    ip: str
    type: str
    servicePort: str
    topicPort: str
    serviceList: List[ServiceName]
    topicList: List[TopicName]


class NetComponent(abc.ABC):
    def __init__(self):
        if NetManager.manager is None:
            raise ValueError("NetManager is not initialized")
        self.manager: NetManager = NetManager.manager
        self.running: bool = False
        self.host_ip: str = self.manager.local_info["ip"]

    def shutdown(self) -> None:
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(NetComponent):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.socket = self.manager.pub_socket
        if topic in self.manager.local_info["topicList"]:
            logger.warning(f"Host {topic} is already registered")
        else:
            self.manager.local_info["topicList"].append(topic)
            logger.info(f'Publisher for topic "{self.topic}" is ready')

    def publish_bytes(self, data: bytes) -> None:
        msg = b''.join([f"{self.topic}:".encode(), b":", data])
        self.manager.submit_task(self.send_bytes_async, msg)

    def publish_dict(self, data: Dict) -> None:
        self.publish_string(dumps(data))

    def publish_string(self, string: str) -> None:
        msg = f"{self.topic}:{string}"
        self.manager.submit_task(self.send_bytes_async, msg.encode())

    def on_shutdown(self) -> None:
        self.manager.local_info["topicList"].remove(self.topic)

    async def send_bytes_async(self, msg: bytes) -> None:
        await self.socket.send(msg)


class Streamer(Publisher):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], Optional[Union[str, bytes, Dict]]],
        fps: int = 45,
    ):
        super().__init__(topic)
        self.running = False
        self.dt: float = 1 / fps
        self.update_func = update_func
        self.manager.submit_task(self.update_loop)
        self.topic_byte = self.topic.encode("utf-8")

    def generate_byte_msg(self) -> bytes:
        return json.dumps(
            {
                "updateData": self.update_func(),
                "time": time.monotonic(),
            }
        ).encode("utf-8")

    async def update_loop(self):
        self.running = True
        last = 0.0
        try:
            while self.running:
                diff = time.monotonic() - last
                if diff < self.dt:
                    await async_sleep(self.dt - diff)
                last = time.monotonic()
                await self.socket.send(
                    b"".join([self.topic_byte, b":", self.generate_byte_msg()])
                )
        except Exception as e:
            logger.error(f"Error when streaming {self.topic}: {e}")


class ByteStreamer(Streamer):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], bytes],
        fps: int = 45,
    ):
        super().__init__(topic, update_func, fps)
        self.update_func: Callable[[], bytes]

    def generate_byte_msg(self) -> bytes:
        return self.update_func()


class Service(NetComponent):
    def __init__(
        self,
        service_name: str,
        callback: Callable[[str], Union[str, bytes, Dict]],
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.callback_func = callback
        self.socket = self.manager.service_socket
        # register service
        self.manager.local_info["serviceList"].append(service_name)
        self.manager.service_list[service_name] = self
        self.on_trigger_events: List[Callable[[str], None]] = []
        logger.info(f'"{self.service_name}" Service is ready')

    async def callback(self, msg: str):
        result = await asyncio.wait_for(
            self.manager.loop.run_in_executor(
                self.manager.executor, self.callback_func, msg
            ),
            timeout=5.0,
        )
        for event in self.on_trigger_events:
            event(msg)
        await self.send(result)

    @abc.abstractmethod
    async def send(self, data: Union[str, bytes, Dict]):
        raise NotImplementedError

    def on_shutdown(self):
        pass


class StringService(Service):
    async def send(self, string: str):
        await self.socket.send_string(string)


class BytesService(Service):
    async def send(self, data: bytes):
        await self.socket.send(data)


class DictService(Service):
    async def send(self, data: Dict):
        await self.socket.send_string(dumps(data))


class NetManager:

    manager = None

    def __init__(
        self, host_ip: IPAddress = "127.0.0.1", host_name: str = "SimPub"
    ) -> None:
        NetManager.manager = self
        self._initialized = False
        self.zmq_context = zmq.asyncio.Context()
        # subscriber
        self.sub_socket_dict: Dict[IPAddress, AsyncSocket] = {}
        # publisher
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:{ServerPort.TOPIC.value}")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:{ServerPort.SERVICE.value}")
        self.service_list: Dict[str, Service] = {}
        # message for broadcasting
        self.local_info: HostInfo = {
            "name": host_name,
            "instance": str(uuid.uuid4()),
            "ip": host_ip,
            "type": "Server",
            "servicePort": str(ServerPort.SERVICE.value),
            "topicPort": str(ServerPort.TOPIC.value),
            "serviceList": [],
            "topicList": [],
        }
        # client info
        self.clients: Dict[IPAddress, HostInfo] = {}
        # start the server in a thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.server_future = self.executor.submit(self.start_event_loop)
        # wait for the loop started
        while not self.running:
            time.sleep(0.01)
        # default service for client registration
        self.register_service = StringService(
            "Register", self.register_client_callback
        )
        self.server_timestamp_service = StringService(
            "GetServerTimestamp", self.get_server_timestamp_callback
        )
        self.client_quit_service = StringService(
            "ClientQuit", self.client_quit_callback
        )
        self.clients_info_service = DictService(
            "GetClientInfo", self.get_clients_info
        )

    def create_socket(self, socket_type: int):
        return self.zmq_context.socket(socket_type)

    def start(self):
        self._initialized = True

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # wait for the start signal
        while not self._initialized:
            time.sleep(0.01)
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)
        self.running = True
        self.loop.run_forever()

    def submit_task(
        self,
        task: Callable,
        *args,
    ) -> Optional[concurrent.futures.Future]:
        if not self.loop:
            return
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def stop_server(self):
        if not self.running:
            return
        if not self.loop:
            return
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError as e:
            logger.error(f"One error occurred when stop server: {e}")
        self.join()

    def join(self):
        self.executor.shutdown(wait=True)

    async def service_loop(self):
        # TODO: restart service loop after an exception is raised
        logger.info("The service is running...")
        while self.running:
            message = await self.service_socket.recv_string()
            if ":" not in message:
                logger.error('Invalid message with no split marker ":"')
                await self.service_socket.send_string("Invalid message")
                continue
            service, request = message.split(":", 1)
            # the zmq service socket is blocked and only run one at a time
            if service in self.service_list.keys():
                try:
                    await self.service_list[service].callback(request)
                except asyncio.TimeoutError:
                    logger.error("Timeout: callback function took too long")
                    await self.service_socket.send_string("Timeout")
                except Exception as e:
                    logger.error(
                        f"One error occurred when processing the Service "
                        f'"{service}": {e}'
                    )
            await async_sleep(0.01)

    async def broadcast_loop(self):
        logger.info(
            f"The Net Manager starts broadcasting at {self.local_info['ip']}"
        )
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # calculate broadcast ip
        local_info = self.local_info
        ip_bin = struct.unpack("!I", socket.inet_aton(local_info["ip"]))[0]
        netmask_bin = struct.unpack("!I", socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack("!I", broadcast_bin))
        address = (broadcast_ip, ServerPort.DISCOVERY.value)
        while self.running:
            local_info = self.local_info  # update local info
            msg = f"SimPub:{local_info['instance']}:{json.dumps(local_info)}"
            _socket.sendto(msg.encode(), address)
            await async_sleep(0.1)
        logger.info("Broadcasting has been stopped")

    def register_client_callback(self, msg: str) -> str:
        client_info: HostInfo = json.loads(msg)
        # NOTE: the client info may be updated so the reference cannot be used
        # NOTE: TypeDict is somehow block if the key is not in the dict
        # if client_info["ip"] not in self.clients:
        self.clients[client_info["ip"]] = client_info
        logger.info(
            f"Host \"{client_info['name']}\" with "
            f"IP \"{client_info['ip']}\" has been registered"
        )
        return "The info has been registered"

    def get_server_timestamp_callback(self, msg: str) -> str:
        return str(time.monotonic())

    def shutdown(self):
        logger.info("Shutting down the server")
        self.running = False
        self.stop_server()
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        logger.info("Server has been shut down")

    def client_quit_callback(self, client_ip: str):
        if client_ip in self.clients:
            del self.clients[client_ip]
            logger.info(f"Client from {client_ip} has quit")
        else:
            logger.warning(f"Client from {client_ip} is not registered")
        return "Client has been removed"

    def get_clients_info(self, msg: str) -> Dict:
        return {ip: info for ip, info in self.clients}


def init_net_manager(host: str) -> NetManager:
    if NetManager.manager is not None:
        return NetManager.manager
    return NetManager(host)
