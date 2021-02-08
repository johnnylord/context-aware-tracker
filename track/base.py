from abc import ABC, abstractmethod


__all__ = [ "TrackState", "TrackAction", "BaseTrack" ]


class TrackState:
    """Helper class containing all track state"""
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    DEAD = 3


class TrackAction:
    HIT = 0
    MISS = 1


class BaseTrack(ABC):
    """Base class for all kind of derived track class

    State transition machine:

                              [hit]
                            +------+
                            v      |
    5 continous at begin  +--------+-+    5 continuous [miss]
            +------------>| Tracked  +-------------+
            |             +----------+             v   +----+
       +----+-----+             ^             +--------++   |
    -->| Tentative|             +-------------+ Lost    |  [miss]
       +----+-----+                  1 [hit]  +----+----+   |
            |                                      |   ^    |
            |             +----------+             |   +----+
            +------------>+  Dead    |<------------+
            1 [miss]      +----------+      exceed max_priority [miss]

    Attributes:
        id (int): id number of the track
        state (int): current state of the track
    """
    MAX_PRIORITY = 30
    ACTIVE_THRESHOLD = 5
    CONFIRM_THRESHOLD = 5

    def __init__(self, id, **kwargs):
        # Public members
        self.id = id
        self.state = TrackState.TENTATIVE
        self.priority = BaseTrack.MAX_PRIORITY

        # Prviate members
        self._hit_count = 1
        self._miss_count = 0
        self._recent_actions = []

    def __str__(self):
        # Determin state
        if self.state == TrackState.TENTATIVE:
            state = "tentative"
        elif self.state == TrackState.TRACKED:
            state = "tracked"
        elif self.state == TrackState.LOST:
            state = "lost"
        elif self.state == TrackState.DEAD:
            state = "dead"
        content = f"{self.__class__.__name__}{self.id} -> '{state}'"
        return content

    def __repr__(self):
        return self.__str__()

    def hit(self):
        self.priority = BaseTrack.MAX_PRIORITY
        # Update private information
        self._hit_count += 1
        self._recent_actions.append(TrackAction.HIT)
        if len(self._recent_actions) > BaseTrack.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update track state
        if (
            self.state == TrackState.TENTATIVE
            and self._hit_count >= BaseTrack.ACTIVE_THRESHOLD
        ):
            self.state = TrackState.TRACKED
        elif (
            self.state == TrackState.TENTATIVE
            and self._hit_count < BaseTrack.ACTIVE_THRESHOLD
        ):
            self.state = TrackState.TENTATIVE
        else:
            self.state = TrackState.TRACKED

    def miss(self):
        self.priority -= 1
        # Update private information
        self._miss_count += 1
        self._recent_actions.append(TrackAction.MISS)
        if len(self._recent_actions) > BaseTrack.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update track state
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DEAD
        elif (
            self.state == TrackState.TRACKED
            and len([ action
                    for action in self._recent_actions
                    if action == TrackAction.MISS ]) >= BaseTrack.CONFIRM_THRESHOLD
        ):
            self.state = TrackState.LOST
        elif (
            self.state == TrackState.LOST
            and self.priority <= 0
        ):
            self.state = TrackState.DEAD

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Predict next state of the track"""
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update internal state of the track"""
        raise NotImplementedError
