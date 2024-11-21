import concurrent.futures
import enum
from typing import List, Dict
from typing import NewType, Callable, TypedDict, Union
import asyncio
from asyncio import sleep as asycnc_sleep
import zmq
import zmq.asyncio
import socket
from socket import AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import struct
import time
import json
from json import dumps
import uuid
from .log import logger
import abc
import concurrent

IPAddress = NewType("IPAddress", str)
TopicName = NewType("TopicName", str)
ServiceName = NewType("ServiceName", str)


class ServerPort(int, enum.Enum):
    # ServerPort and ClientPort need using .value to get the port number
    # which is not supposed to be used in this way
    DISCOVERY = 7720
    SERVICE = 7721
    TOPIC = 7722


class ClientPort(int, enum.Enum):
    DISCOVERY = 7720
    SERVICE = 7730
    TOPIC = 7731


class HostInfo(TypedDict):
    name: str
    ip: IPAddress
    topics: List[TopicName]
    services: List[ServiceName]


class SimPubClient:
    def __init__(self, client_info: HostInfo) -> None:
        self.manager: NetManager = NetManager.manager
        self.info = client_info
        self.req_socket: zmq.Socket = self.manager.zmq_context.socket(zmq.REQ)
        self.sub_socket: zmq.Socket = self.manager.zmq_context.socket(zmq.SUB)
        self.req_socket.connect(
            f"tcp://{client_info['ip']}:{ClientPort.SERVICE.value}")
        self.sub_socket.connect(
            f"tcp://{client_info['ip']}:{ClientPort.TOPIC.value}")


class Communicator(abc.ABC):
    def __init__(self):
        self.running: bool = False
        self.manager: NetManager = NetManager.manager
        self.host_ip: str = self.manager.local_info["ip"]
        self.host_name: str = self.manager.local_info["host"]

    def shutdown(self):
        self.running = False
        self.on_shutdown()

    @abc.abstractmethod
    def on_shutdown(self):
        raise NotImplementedError


class Publisher(Communicator):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.socket = self.manager.pub_socket
        if topic in self.manager.local_info["topics"]:
            logger.warning(f"Host {topic} is already registered")
        else:
            self.manager.local_info["topics"].append(topic)
        logger.info(f"Publisher for topic \"{self.topic}\" is ready")

    def publish(self, data: Dict):
        msg = f"{self.topic}:{dumps(data)}"
        self.manager.submit_task(self.send_msg_async, msg)

    def publish_string(self, string: str):
        self.manager.submit_task(self.send_msg_async, f"{self.topic}:{string}")

    def on_shutdown(self):
        super().on_shutdown()

    async def send_msg_async(self, msg: str):
        await self.socket.send_string(msg)


class Streamer(Publisher):
    def __init__(
        self,
        topic: str,
        update_func: Callable[[], Dict],
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
                    await asycnc_sleep(self.dt - diff)
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

    def generate_byte_msg(self) -> bytes:
        return self.update_func()


class Service(Communicator):
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
        self.manager.local_info["services"].append(service_name)
        self.manager.service_list[service_name] = self
        logger.info(f'"{self.service_name}" Service is ready')

    async def callback(self, msg: str):
        result = await asyncio.wait_for(
            self.manager.loop.run_in_executor(
                self.manager.executor, self.callback_func, msg
            ),
            timeout=5.0,
        )
        await self.send(result)

    @abc.abstractmethod
    async def send(self, string: str):
        raise NotImplementedError

    def on_shutdown(self):
        return super().on_shutdown()


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
        self.sub_socket_dict: Dict[IPAddress, zmq.Socket] = {}
        # publisher
        self.pub_socket = self.zmq_context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host_ip}:{ServerPort.TOPIC.value}")
        # service
        self.service_socket = self.zmq_context.socket(zmq.REP)
        self.service_socket.bind(f"tcp://{host_ip}:{ServerPort.SERVICE.value}")
        self.service_list: Dict[str, Service] = {}
        # message for broadcasting
        self.local_info = HostInfo()
        self.local_info["host"] = host_name
        self.local_info["ip"] = host_ip
        self.local_info["topics"] = []
        self.local_info["services"] = []
        # client info
        self.on_client_registered: List[Callable[[HostInfo], None]] = list()
        self.clients: Dict[IPAddress, SimPubClient] = {}
        # setting up thread pool
        self.running: bool = True
        self.loop: asyncio.AbstractEventLoop = None
        # start the server in a thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.server_future = self.executor.submit(self.start_event_loop)
        while self.loop is None:
            time.sleep(0.01)

    def start(self):
        self._initialized = True

    def start_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # wait for the start signal
        while not self._initialized:
            time.sleep(0.01)
        # default service for client registration
        self.register_service = StringService(
            "Register", self.register_client_callback
        )
        self.server_timestamp_service = StringService(
            "GetServerTimestamp", self.get_server_timestamp_callback
        )
        # default task for client registration
        self.submit_task(self.broadcast_loop)
        self.submit_task(self.service_loop)
        self.loop.run_forever()

    def submit_task(self, task: Callable, *args) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(task(*args), self.loop)

    def stop_server(self):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.loop.stop(), self.loop)
        self.executor.shutdown(wait=True)

    def join(self):
        self.executor.shutdown(wait=True)

    async def service_loop(self):
        # try:
        logger.info("The service is running...")
        while self.running:
            message = await self.service_socket.recv_string()
            if ":" not in message:
                logger.error("Invalid message with no spliter \":\"")
                await self.service_socket.send_string("Invalid message")
                continue
            service, request = message.split(":", 1)
            # the zmq service socket is blocked and only run one at a time
            if service in self.service_list.keys():
                try:
                    await self.service_list[service].callback(request)
                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout: callback function took too long to execute"
                    )
                    await self.service_socket.send_string("Timeout")
                except Exception as e:
                    logger.error(
                        f"One error ocurred when processing the Service "
                        f'"{service}": {e}'
                    )
            await asycnc_sleep(0.01)

    async def broadcast_loop(self):
        logger.info("The server is broadcasting...")
        # set up udp socket
        _socket = socket.socket(AF_INET, SOCK_DGRAM)
        _socket.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        _id = str(uuid.uuid4())
        # calculate broadcast ip
        local_info = self.local_info
        ip_bin = struct.unpack("!I", socket.inet_aton(local_info["ip"]))[0]
        netmask_bin = struct.unpack("!I", socket.inet_aton("255.255.255.0"))[0]
        broadcast_bin = ip_bin | ~netmask_bin & 0xFFFFFFFF
        broadcast_ip = socket.inet_ntoa(struct.pack("!I", broadcast_bin))
        while self.running:
            local_info = self.local_info  # update local info
            msg = f"SimPub:{_id}:{json.dumps(local_info)}"
            _socket.sendto(
                msg.encode(), (broadcast_ip, ServerPort.DISCOVERY.value)
            )
            await asycnc_sleep(0.1)
        logger.info("Broadcasting has been stopped")

    def register_client_callback(self, msg: str) -> str:
        # NOTE: something woring with sending message, but it solved somehow
        client_info: HostInfo = json.loads(msg)
        # NOTE: the client info may be updated so the reference cannot be used
        # NOTE: TypeDict is somehow block if the key is not in the dict
        if client_info["ip"] not in self.clients:
            self.clients[client_info["ip"]] = SimPubClient(client_info)
        for callback in self.on_client_registered:
            callback(self.clients[client_info["ip"]])
        logger.info(
            f"Host \"{client_info['name']}\" with"
            f"IP \"{client_info['ip']}\" has been registered"
        )
        return "The info has been registered"

    def get_server_timestamp_callback(self, msg: str) -> str:
        return str(time.monotonic())

    def shutdown(self):
        logger.info("Shutting down the server")
        self.pub_socket.close(0)
        self.service_socket.close(0)
        for sub_socket in self.sub_socket_dict.values():
            sub_socket.close(0)
        self.running = False
        logger.info("Server has been shut down")


def init_net_manager(host: str) -> NetManager:
    if NetManager.manager is not None:
        return NetManager.manager
    return NetManager(host)
