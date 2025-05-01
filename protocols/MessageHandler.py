"""
Message protocol which handles all classical messages between nodes.
"""
import operator
from enum import Enum, auto
from functools import reduce

from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.components.component import Message, Port


class MessageType(Enum):

    # entanglement signals
    GEN_ENTANGLE_READY = auto()
    ENTANGLED = auto()
    ENTANGLED_QUBIT_LOST = auto()
    GEN_ENTANGLE_SUCCESS = auto()
    ENTANGLED_SUCCESS = auto()
    # entanglement concurrent signals
    RE_ENTANGLE_CONCURRENT = auto()
    # re-entanglement signals
    RE_ENTANGLE = auto()
    RE_ENTANGLE_READY = auto()
    RE_ENTANGLE_READY_REMOTE = auto()
    RE_ENTANGLE_READY_SOURCE = auto()
    RE_ENTANGLE_FROM_UPPER_LAYER = auto()
    RE_ENTANGLE_QUBIT_LOST = auto()
    # purification signals
    PURIFICATION_START = auto()
    PURIFICATION_RESULT = auto()
    PURIFICATION_TARGET_MET = auto()
    PURIFICATION_NEED_SHUTDOWN = auto()
    PURIFICATION_SUCCESS = auto()
    # verification signals
    VERIFICATION_REQUEST = auto()
    VERIFICATION_READY = auto()
    VERIFICATION_START = auto()
    VERIFICATION_RESULT = auto()
    # swap signals
    SWAP_NEED = auto()
    SWAP_READY = auto()
    SWAP_APPLY_CORRECTION = auto()
    SWAP_APPLY_CORRECTION_SUCCESS = auto()
    SWAP_SUCCESS = auto()
    SWAP_FAILED = auto()
    # transport signals
    TRANSPORT_REQUEST = auto()
    TRANSPORT_READY = auto()
    TRANSPORT_APPLY_CORRECTION = auto()
    TRANSPORT_APPLY_CORRECTION_SUCCESS = auto()
    TRANSPORT_SUCCESS = auto()
    # termination signals
    ENTANGLEMENT_HANDLER_FINISHED = auto()
    PURIFICATION_FINISHED = auto()
    VERIFICATION_FINISHED = auto()
    SECURITY_VERIFICATION_FINISHED = auto()
    SWAP_FINISHED = auto()
    TRANSPORT_FINISHED = auto()
    # security signals
    SECURITY_TRANSPORT_START = auto()
    SECURITY_TRANSPORT_QUBIT = auto()
    SECURITY_VERIFICATION_REQUEST = auto()
    SECURITY_VERIFICATION_READY = auto()
    SECURITY_VERIFICATION_START = auto()
    SECURITY_VERIFICATION_RESULT = auto()
    # CHSH verification
    CHSH_BASIS_ANNOUNCEMENT = auto()
    CHSH_MEASUREMENT_REQUEST = auto()
    # For basis choice coordination
    CHSH_MEASUREMENT_RESULT = auto()
    # For outcome exchange
    CHSH_FINAL_RESULTS = auto()
    CHSH_FINISHED = auto()
    # GHZ verification
    GHZ_BASIS_ANNOUNCEMENT = auto()
    GHZ_MEASUREMENT_REQUEST = auto()
    # For basis choice coordination
    GHZ_MEASUREMENT_RESULT = auto()
    # For outcome exchange
    GHZ_FINAL_RESULTS = auto()
    GHZ_FINISHED = auto()

class MessageHandler(NodeProtocol):
    """
    A protocol that handles all classical messages between nodes.
    """

    def __init__(self, node, name, cc_ports):
        super().__init__(node=node, name=name)
        self.node = node
        self.cc_ports = cc_ports
        self.add_signals()

    def send_message(self, message_type, dest, data):
        """
        Send a message to the destination node.
        :param message_type: message signal type
        :param dest: destination node name
        :param data: message data
        :return:
        """
        cport = self.cc_ports[dest]
        cport.tx_output(Message(data, header=message_type))

    def add_signals(self):
        # add all signals from enum
        for signal in MessageType:
            self.add_signal(signal)

    # def send_signals(self, signal, msg):
    #     """
    #     Emit a signal from MessageHandler.
    #     :param signal: signal to emit
    #     :param msg: message data
    #     """
    #     self.node.send_signal(signal, msg)

    def run(self):
        expression = reduce(operator.or_, [self.await_port_input(port) for port in self.cc_ports.values()])
        while True:
            # yield until a message is received
            expr = yield expression
            for event in expr.triggered_events:
                port = event.source
                message = port.rx_input()
                # TODO why we have None here??? possible reason is due to reset?
                #  Temp fix is ignore the None
                # if message is None:
                #     continue
                for msg in message.items:
                    self.send_signal(message.meta['header'], msg)
            # if message.header == MessageType.ENTANGLED:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_NEED:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_READY:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_RESULT:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.SWAP_FAILED:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.CORRECTION_SUCCESS:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.RE_ENTANGLE:
            #     self.node.qmemory.put(message.data)
            # else:
            #     raise ValueError(f"Unknown message type: {message.header}")

    def reset(self):
        for port in self.cc_ports.values():
            port.reset()