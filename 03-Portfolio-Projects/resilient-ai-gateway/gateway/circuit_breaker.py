import time
from enum import Enum

class BreakerState(Enum):
    CLOSED = "CLOSED"         # System healthy, forwarding requests to primary provider
    OPEN = "OPEN"             # System faulted, rerouting immediately to backup
    HALF_OPEN = "HALF_OPEN"   # Testing primary with single check probes

class StatefulCircuitBreaker:
    """
    Stateful Circuit Breaker implementing standard fault-tolerance transitions.
    Protects downstream applications from model provider outages by instantly
    hot-swapping routes to alternative endpoints.
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.state = BreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_state_change = time.time()

    def get_route(self) -> BreakerState:
        """
        Evaluates the current state of the circuit and handles transitions.
        """
        current_time = time.time()

        if self.state == BreakerState.OPEN:
            # If recovery timeout has expired, transition to HALF_OPEN to check primary
            if current_time - self.last_state_change > self.recovery_timeout:
                self.transition_to(BreakerState.HALF_OPEN)
                
        return self.state

    def log_success(self):
        """
        Logs a successful upstream response, resetting failure metrics.
        """
        self.failure_count = 0
        if self.state == BreakerState.HALF_OPEN:
            self.transition_to(BreakerState.CLOSED)

    def log_failure(self):
        """
        Logs a failed upstream response (e.g., 503, 500, or persistent timeouts).
        Trips the circuit to OPEN if threshold is crossed.
        """
        self.failure_count += 1
        print(f"❌ Upstream failure logged. Count: {self.failure_count}/{self.failure_threshold}")
        
        if self.state in (BreakerState.CLOSED, BreakerState.HALF_OPEN):
            if self.failure_count >= self.failure_threshold:
                self.transition_to(BreakerState.OPEN)

    def transition_to(self, new_state: BreakerState):
        """
        Triggers a state transition and logs the architectural shift.
        """
        print(f"🔄 [Circuit Breaker] Transitioning state: {self.state.value} ➡️ {new_state.value}")
        self.state = new_state
        self.last_state_change = time.time()
        
        # Reset count on transitions
        if new_state == BreakerState.CLOSED:
            self.failure_count = 0
