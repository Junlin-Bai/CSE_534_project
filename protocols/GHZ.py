from collections import defaultdict

import numpy as np
from netsquid.components.qprocessor import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.qubits.operators import Z, X, Y, Operator
from netsquid.qubits import qubitapi as qapi

import netsquid as ns
from sympy.physics.units import years

from protocols.MessageHandler import MessageType
from utils import Logging, SignalMessages
from utils.ClassicalMessages import ClassicalMessage

import random


############################
# The ExampleEntanglement Protocol
############################


class GHZProtocol(NodeProtocol):
    """
    GHZ
    :param: sample_size_rate: the rate of sampling size for all entangled pairs.
    """

    def __init__(self,
                 node,
                 name,
                 entangle_node,
                 qubit_ready_protocols,
                 setting_list,
                 total_pairs,
                 sample_size_rate,
                 alpha,
                 cc_message_handler,
                 logger,
                 delta=0.1
                 ):

        if entangle_node is None:
            raise ValueError("Entangle node must be specified.")

        super().__init__(node=node, name=name)
        # parameter for projection
        self.alpha = alpha
        # self.epsilon = alpha / np.sqrt(2)
        self.delta = delta
        self.theta_th = 2 * np.sqrt(2) - 5 * np.sqrt(2) * (alpha / 3)
        # self.N = int(np.min(
        #     [3 / (self.delta * alpha ** 2), 16 / alpha ** 2 * np.log(2 / (1 - (1 - self.delta / 2) ** (1 / 4)))]))
        # self.N_4 = 4 * self.N

        # keep track of when we started the protocol. we can use this avoid process old data
        self.start_time = None
        self.entangled_node = entangle_node
        self.logger = logger
        self.sample_pairs_needed = int(total_pairs * sample_size_rate)
        self.total_pairs = total_pairs
        self.cc_message_handler = cc_message_handler
        self.entangled_pairs = {}  # key = mem_pos : fid?
        self.GHZ_done = False
        self.qubit_ready_protocols = qubit_ready_protocols
        # determine who will be sending start measurement
        self.is_source = None
        self.s_value = None
        self.measurement_poses = []
        # GHZ parameter
        # Prepare a list of measurement settings, ensuring each of the four combinations is used equally
        self.settings_list = setting_list
        # Initialize a dictionary to store correlation outcomes for each measurement setting combination.
        self.measure_result = []  # (setting, value)

        self.add_signal(MessageType.GHZ_FINISHED)

    def run(self):
        self.logger.info(f"GHZ Started {self.name} -> Node {self.node.name}", color="cyan")
        entangle_signal = self.await_signal(self.qubit_ready_protocols, signal_label=MessageType.ENTANGLED_SUCCESS)
        measurement_signal = (
                self.await_signal(self.cc_message_handler, signal_label=MessageType.GHZ_MEASUREMENT_RESULT)
                | self.await_signal(self.cc_message_handler, signal_label=MessageType.GHZ_FINAL_RESULTS)
                | self.await_signal(self.cc_message_handler, signal_label=MessageType.GHZ_MEASUREMENT_REQUEST))

        self.start_time = sim_time()
        while True:
            # wait entangle signal from EH
            # if we have enough sample EPR, we start GHZ measurement
            # once we done we signal we are done
            expr = entangle_signal | measurement_signal

            yield expr

            if expr.first_term.value:
                # handle the entangle signal from the entanglement handler
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                    except Exception as e:
                        self.logger.info(f"GHZ {self.name} -> "
                                         f"Node {self.node.name} failed to get signal: {e}",
                                         color="red")
                        continue
                    result: SignalMessages.EntangleSuccessSignalMessage = ready_signal.result
                    if result.timestamp < self.start_time:
                        continue
                    if ready_signal.label == MessageType.ENTANGLED_SUCCESS:
                        self.logger.info(f"GHZ {self.name} -> "
                                         f"Node {self.node.name} received entangle signal\n"
                                         f"\tCount {len(result.mem_pos)}", color="blue")
                        if result.entangle_node != self.entangled_node:
                            continue
                        self.handle_entangle_signal(result)
            elif expr.second_term.value:
                # case of GHZ result
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                        result: ClassicalMessage = ready_signal.result
                        if result is None:
                            continue
                    except Exception as e:
                        self.logger.error(f"GHZ {self.name} -> "
                                          f"Node {self.node.name} error processing classical message: {e}",
                                          color="red")
                        continue
                    if isinstance(result, ClassicalMessage):
                        if result.from_node != self.entangled_node:
                            # we do not care about the message from other nodes
                            continue
                        if result.data.timestamp < self.start_time:
                            # discard race condition where previous round info just arrived
                            continue
                    if ready_signal.label == MessageType.GHZ_MEASUREMENT_RESULT:
                        result: SignalMessages.GHZMeasurementResultSignalMessage = result.data
                        self.proccess_GHZ_measurement(result)
                        if self.check_GHZ_done():
                            self.send_signal(MessageType.GHZ_FINISHED, {"s_value": self.s_value,
                                                                         "theta": self.theta_th,
                                                                         "entangled_pairs": self.entangled_pairs,
                                                                         "finish_time": sim_time()})
                            break
                    elif ready_signal.label == MessageType.GHZ_MEASUREMENT_REQUEST:
                        # case of bob received alice request to start measurement
                        result: SignalMessages.GHZStartMeasurementSignalMessage = result.data
                        self.measurement_poses = result.mem_pos
                        self.logger.info(f"GHZ {self.name} -> Got Measurement Request\n"
                                         f"\tPos Count: {len(self.measurement_poses)}", color="yellow")
                        yield from self.start_GHZ_measurement(self.measurement_poses)
            if self.is_source and self.check_GHZ_ready():
                # take the qubits as measurement needs
                measurement_qubits = list(self.entangled_pairs.keys())[:self.sample_pairs_needed]
                self.logger.info(f"GHZ {self.name} -> Start measurement\n"
                                 f"\tPos Count: {len(measurement_qubits)}", color="green")

                self.measurement_poses = measurement_qubits
                # send this to Bob so it knows what to remove
                self.cc_message_handler.send_message(MessageType.GHZ_MEASUREMENT_REQUEST,
                                                     self.entangled_node,
                                                     data=ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.entangled_node,
                                                         data=SignalMessages.GHZStartMeasurementSignalMessage(
                                                             mem_pos=measurement_qubits,
                                                         )
                                                     )
                                                     )
                yield from self.start_GHZ_measurement(measurement_qubits)
            if self.is_source and self.check_GHZ_done():
                self.send_signal(MessageType.GHZ_FINISHED, {"s_value": self.s_value,
                                                             "theta": self.theta_th,
                                                             "entangled_pairs": self.entangled_pairs,
                                                             "finish_time": sim_time()})
                break

    def handle_entangle_signal(self, result: SignalMessages.EntangleSuccessSignalMessage):
        # store the entangle info
        for mem_pos, fid in zip(result.mem_pos, result.fidelity):
            if mem_pos in self.measurement_poses:
                continue
            self.entangled_pairs[mem_pos] = fid
            self.is_source = result.is_source

    def check_GHZ_done(self):
        if self.s_value is not None and len(self.entangled_pairs) >= self.total_pairs - self.sample_pairs_needed:
            return True
        else:
            return False

    def start_GHZ_measurement(self, measurement_qubits):
        # remove from entangle pairs
        for i in measurement_qubits:
            if i in self.entangled_pairs:
                self.entangled_pairs.pop(i)

        # start measurement

        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        for index, mem_pos in enumerate(measurement_qubits):
            setting = self.settings_list[index]
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(mem_pos, skip_noise=False)
            # self.logger.info(f"GHZ {self.name} -> Measure\n"
            #                  f"Pos: {mem_pos}\n"
            #                  f"Qubit {qubit}")
            if setting == "A0":
                outcome_a = self.measure_in_A_0_basis(qubit)
                self.measure_result.append(("A0", outcome_a))
                outcome_b = self.measure_in_B_0_basis(qubit)
                self.measure_result.append(("B0", outcome_b))
            elif setting == "A1":
                outcome_a = self.measure_in_A_1_basis(qubit)
                self.measure_result.append(("A1", outcome_a))
                outcome_b = self.measure_in_B_1_basis(qubit)
                self.measure_result.append(("B1", outcome_b))

        self.GHZ_done = True
        # now sent the information to Alice if we are bob
        if not self.is_source:
            self.cc_message_handler.send_message(
                MessageType.GHZ_MEASUREMENT_RESULT,
                self.entangled_node,
                ClassicalMessage(
                    self.node.name,
                    self.entangled_node,
                    SignalMessages.GHZMeasurementResultSignalMessage(self.measure_result))

            )

    def proccess_GHZ_measurement(self, message):
        """
        Process the GHZ measurement result from Bob
        :param message: SignalMessages.GHZMeasurementResultSignalMessage
        """
        res = message.measurement_results
        corr_map = defaultdict(list)
        for (op_a, outcome_a), (op_b, outcome_b) in zip(self.measure_result, res):
            corr_key = f"{op_a}{op_b}"
            corr_map[corr_key].append(outcome_a ^ outcome_b)
        # take average
        s1 = 0
        s2 = 0
        n1 = 0  # A1B1
        n2 = 0  # A0B0
        for k, v in corr_map.items():
            self.logger.info(f"GHZ {self.name} -> Measure Value\n"
                             f"\tOP {k}\n"
                             f"\tValue {v}", color="yellow")
            if k == "A1B1":
                s1 += np.sum(v)
                n1 += len(v)
            else:
                flipped_v = [1 - x for x in v]
                s2 += np.sum(flipped_v)
                n2 += len(v)
        if n1 + n2 > 0:
            s = (s1 + s2) / (n1 + n2)
        else:
            s = 0  # raise Exception("No data")
        self.s_value = s
        # send s to bob
        # self.cc_message_handler.send_message(MessageType.GHZ_FINAL_RESULTS,
        #                                      self.entangled_node,
        #                                      data=SignalMessages.GHZFinalResultSignalMessage(
        #                                          s_value=s
        #                                      )
        #                                      )
        #

    def check_GHZ_ready(self):
        # check wether we are ready for the GHZ measurement
        if len(self.entangled_pairs) >= self.sample_pairs_needed and self.GHZ_done is False:
            # For each entangled pair, coordinate measurement settings via classical communication,
            return True
        return False

    def reset(self):
        self.s_value = None
        self.measurement_poses = []
        self.GHZ_done = False
        self.entangled_pairs = {}
        self.measure_result = []
        self.start_time = None
        super().reset()

    @staticmethod
    def measure_in_A_0_basis(qubit):
        """
        x as observable
        Measure a qubit in the X basis.

        Parameters:
            qubit: The identifier or reference of the qubit.

        Returns:
            0 if the measurement outcome is 0,
            1 if the measurement outcome is 1.
        """
        outcome, _ = qapi.measure(qubit, observable=X)
        return 0 if outcome == 0 else 1

    @staticmethod
    def measure_in_A_1_basis(qubit):
        """
        Measure a qubit in Bob's B1 basis using a custom observable (x as observable).
        x as observable
        Measure a qubit in the X basis.

        Parameters:
            qubit: The identifier or reference of the qubit.

        Returns:
            0 if the measurement outcome is 0,
            1 if the measurement outcome is 1.
        """
        outcome, _ = qapi.measure(qubit, observable=Y)
        return 0 if outcome == 0 else 1

    @staticmethod
    def measure_in_B_0_basis(qubit):
        """
        Measure a qubit in Bob's B1 basis using a custom observable (y as observable).
        y as observable
        Measure a qubit in the Y basis.

        Parameters:
            qubit: The identifier or reference of the qubit.

        Returns:
            0 if the measurement outcome is 0,
            1 if the measurement outcome is 1.
        """
        outcome, _ = qapi.measure(qubit, observable=X)
        return 0 if outcome == 0 else 1

    @staticmethod
    def measure_in_B_1_basis(qubit):
        """
        Measure a qubit in Bob's B1 basis using a custom observable (y as observable).
        y as observable
        Measure a qubit in the Y basis.

        Parameters:
            qubit: The identifier or reference of the qubit.

        Returns:
            0 if the measurement outcome is 0,
            1 if the measurement outcome is 1.
        """
        outcome, _ = qapi.measure(qubit, observable=Y)
        return 0 if outcome == 0 else 1